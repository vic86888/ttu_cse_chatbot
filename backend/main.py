from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
from typing import AsyncGenerator
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager

# RAG 相關套件
from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# 全域變數
DB_DIR = "../storage/chroma"
COLL_NAME = "campus_rag"
EVENT_KEYWORDS = [
    "新聞", "消息", "news", "最新",
    "最近", "活動", "說明會", "講座", "論壇", "營隊", "徵才"
]
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# 初始化 reranker 和 chain（啟動時載入）
reranker = None
rag_chain = None

def rerank_docs(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """用 cross-encoder 對候選文件重新排序"""
    if not docs:
        return []
    
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    
    scored = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    
    out: list[Document] = []
    for doc, s in scored[:top_n]:
        md = dict(doc.metadata) if doc.metadata else {}
        md["rerank_score"] = float(s)
        out.append(Document(page_content=doc.page_content, metadata=md))
    
    return out

def make_scored_retriever(vdb, k: int = 10):
    """建立包含分數的檢索器"""
    k_retrieve = max(k * 4, 20)
    
    def _retrieve(query: str):
        def as_docs(pairs):
            out = []
            for doc, score in pairs:
                md = dict(doc.metadata) if doc.metadata else {}
                md["relevance"] = float(score)
                out.append(Document(page_content=doc.page_content, metadata=md))
            return out
        
        q = (query or "").lower()
        prefer_news = any(kw in q for kw in EVENT_KEYWORDS)
        
        docs: list[Document] = []
        if prefer_news:
            news_pairs = vdb.similarity_search_with_relevance_scores(
                query, k=k_retrieve, filter={"content_type": "news"}
            )
            docs.extend(as_docs(news_pairs))
            
            if len(docs) < k_retrieve:
                more_pairs = vdb.similarity_search_with_relevance_scores(query, k=k_retrieve)
                docs.extend(as_docs(more_pairs))
            
            # 去重
            seen = set()
            uniq = []
            for d in docs:
                md = d.metadata or {}
                key = (
                    ("article", md.get("source"), md.get("article_id"))
                    if md.get("article_id")
                    else ("row", md.get("source"), md.get("idx"))
                )
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(d)
            docs = uniq
        else:
            pairs = vdb.similarity_search_with_relevance_scores(query, k=k_retrieve)
            docs = as_docs(pairs)
        
        # 重新排序
        docs = rerank_docs(query, docs, top_n=k)
        return docs
    
    return RunnableLambda(_retrieve)

def build_chain():
    """建立 RAG chain"""
    # 1) LLM
    llm = ChatOllama(
        model="qwen3:latest",
        temperature=0,
    )
    
    # 2) 提示詞
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "你是大同大學資工系問答機器人。\n"
        "學年等於民國紀年,114學年就是2025年。"
        "你會得到跟問題相關的文件,你只依據提供的文件內容回答問題,"
        "若無法從文件中找到答案,請清楚說明。\n\n"
        "請以繁體中文作答，並使用 Markdown 格式來組織答案：\n"
        "- 使用 ## 二級標題來分隔不同主題\n"
        "- 使用列表 (- 或 1.) 來呈現多個項目\n"
        "- 使用 **粗體** 來強調重要資訊\n"
        "- 使用程式碼區塊 ```語言 來展示程式碼\n"
        "- 使用表格來呈現結構化資料\n\n"
        "{context}"),
        ("human", "{input}")
    ])
    
    # 3) stuff chain
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    # 4) 向量庫 & 檢索器
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )
    
    scored_retriever = make_scored_retriever(vectordb, k=10)
    retriever_runnable = itemgetter("input") | scored_retriever
    
    # 5) RAG 鏈
    rag_chain = create_retrieval_chain(retriever_runnable, doc_chain)
    return rag_chain

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    global reranker, rag_chain
    print("🔄 載入 reranker 模型...")
    reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")
    print("🔄 建立 RAG chain...")
    rag_chain = build_chain()
    print("✅ 模型載入完成！")
    yield
    print("👋 關閉應用...")

app = FastAPI(title="TTU CSE Chatbot API", lifespan=lifespan)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite 預設埠
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "TTU CSE Chatbot API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_ready": rag_chain is not None}

async def generate_stream(message: str) -> AsyncGenerator[str, None]:
    """生成 SSE 串流回應（使用 RAG）"""
    try:
        # 加上當前時間
        now = datetime.now(ZoneInfo("Asia/Taipei"))
        roc_year = now.year - 1911
        today_roc = f"{roc_year}年{now.month}月{now.day}日"
        timestamped_q = f"[當前時間: {today_roc}] {message}"
        
        # 執行 RAG 查詢
        result = await asyncio.to_thread(
            rag_chain.invoke,
            {"input": timestamped_q}
        )
        
        response_text = result["answer"]
        
        # 逐字串流
        for char in response_text:
            yield f"data: {json.dumps({'content': char, 'done': False}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)
        
        # 發送完成訊號
        yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
        
    except Exception as e:
        error_msg = f"抱歉，處理您的問題時發生錯誤：{str(e)}"
        yield f"data: {json.dumps({'content': error_msg, 'done': True, 'error': True}, ensure_ascii=False)}\n\n"

@app.get("/api/chat/stream")
async def chat_stream(message: str):
    """SSE 串流聊天端點"""
    return StreamingResponse(
        generate_stream(message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/api/chat")
async def chat(request: dict):
    """非串流聊天端點（使用 RAG）"""
    try:
        message = request.get("message", "")
        
        # 加上當前時間
        now = datetime.now(ZoneInfo("Asia/Taipei"))
        roc_year = now.year - 1911
        today_roc = f"{roc_year}年{now.month}月{now.day}日"
        timestamped_q = f"[當前時間: {today_roc}] {message}"
        
        # 執行 RAG 查詢
        result = await asyncio.to_thread(
            rag_chain.invoke,
            {"input": timestamped_q}
        )
        
        return {
            "response": result["answer"],
            "sources": [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance": doc.metadata.get("relevance"),
                    "rerank_score": doc.metadata.get("rerank_score")
                }
                for doc in result.get("context", [])
            ]
        }
    except Exception as e:
        return {"response": f"抱歉，處理您的問題時發生錯誤：{str(e)}", "error": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
