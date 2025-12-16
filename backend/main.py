from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
import re
from typing import AsyncGenerator
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager

# RAG / LangChain 相關
from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langsmith import traceable

# ========= 全域設定 =========
# 注意：這裡保留原本後端用的相對路徑
DB_DIR = "../storage/chroma"
COLL_NAME = "campus_rag"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# 全域物件（啟動時建立）
reranker: CrossEncoder | None = None
chains: dict | None = None  # {"rag": ..., "rewrite": ...}

# ========= Rerank / Retriever =========
def rerank_docs(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """用 cross-encoder 對候選文件重新排序，只保留前 top_n。"""
    global reranker
    if not docs:
        return []
    if reranker is None:
        raise RuntimeError("reranker 尚未初始化")

    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)  # numpy array

    scored = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    out: list[Document] = []
    for doc, s in scored[:top_n]:
        md = dict(doc.metadata) if doc.metadata else {}
        md["rerank_score"] = float(s)
        out.append(Document(page_content=doc.page_content, metadata=md))

    return out


def make_scored_retriever(vdb, k: int = 10):
    """Chroma + relevance score + CrossEncoder rerank"""
    # 先抓比較多，再給 reranker 挑前 k
    k_retrieve = max(k * 4, 100)

    def _retrieve(query: str):
        def as_docs(pairs):
            out = []
            for doc, score in pairs:
                md = dict(doc.metadata) if doc.metadata else {}
                md["relevance"] = float(score)
                out.append(Document(page_content=doc.page_content, metadata=md))
            return out

        pairs = vdb.similarity_search_with_relevance_scores(
            query,
            k=k_retrieve,
        )
        docs = as_docs(pairs)

        # 用 cross-encoder 重新排序，只保留前 k 個
        docs = rerank_docs(query, docs, top_n=k)
        return docs

    return RunnableLambda(_retrieve).with_config({
        "run_name": "ChromaRetriever+Reranker",
        "tags": ["retriever", "chroma", "with-scores", "rerank"],
        "metadata": {"k": k},
    })


# ========= Query Rewriter / RAG Chain =========
def build_chain():
    """建立兩條鏈：rewrite_chain（處理時間） + rag_chain（真正回答）"""

    # 1) LLM
    llm = ChatOllama(
        model="qwen3:latest",
        temperature=0,
    ).with_config({
        "run_name": "Ollama-LLM",
        "tags": ["ollama", "qwen3", "local"],
        "metadata": {"provider": "ollama"},
    })

    # 1.5) ✨ 時間相關 Query Rewriter
    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一個查詢改寫器。\n"
            "你知道現在時間是：{now}（時區：Asia/Taipei）。\n"
            "目前學期：{acad_term}\n\n"
            "學年等於民國紀年,114學年就是2025年。"
            "2025年8月1日~2026年7月31日都是114學年以此類推。\n"
            "請閱讀使用者的問題，將其中的「相對時間」"
            "（例如：今天、明天、後天、這週、下週、上週、上個月、下個月、最近幾天、這學期、下學期、今年、明年等）\n"
            "換算成「明確的日期或年月」後，改寫成一個新的問題句子。\n"
            "規則：\n"
            "1. 如果問題中沒有相對時間，就原封不動輸出原始問題。\n"
            "2. 一律使用西元年份（例如：2025年10月），不要使用民國年（例如：民國114年）。\n"
            "3. 不要加入「學年度」「學年」「學期」等字眼，除非使用者原本問題就有。\n"
            "4. 僅輸出改寫後的問題，不要任何解釋、不要加前綴、不要多行說明。\n"
        ),
        ("human", "{query}"),
    ]).with_config({
        "run_name": "TemporalQueryRewriter",
        "tags": ["query-rewrite", "temporal"],
    })

    rewrite_chain = (rewrite_prompt | llm | StrOutputParser()).with_config({
        "run_name": "RewriteChain",
        "tags": ["chain", "rewrite"],
    })

    # 2) 提示詞
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "現在時間：{now}\n\n"
        "目前學期：{acad_term}\n\n"
        "學年等於民國紀年,114學年就是2025年。"
        "2025年8月1日~2026年7月31日都是114學年以此類推。\n"
        "你是大同大學資工系問答機器人。\n"
        "你會得到跟問題相關的文件以及系統提供的資訊（如現在時間、目前學期）,你只依據提供的文件內容回答問題,"
        "回答結尾需附上回答時參考資料的來源網址，若沒有則附上來源文件。"
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

    # 3) combine documents chain
    doc_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    ).with_config({
        "run_name": "StuffDocumentsChain",
        "tags": ["chain", "stuff"],
    })

    # 4) 向量庫 & 檢索器（含分數）
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},  # 若沒有 GPU 可改成 "cpu"
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    scored_retriever = make_scored_retriever(vectordb, k=10)
    # create_retrieval_chain 期望 retriever 接收整個 input dict
    retriever_runnable = itemgetter("input") | scored_retriever

    # 5) RAG 鏈（retriever + combine_docs）
    rag_chain = create_retrieval_chain(retriever_runnable, doc_chain).with_config({
        "run_name": "CampusRAG",
        "tags": ["campus-rag", "api"],
    })

    return {
        "rag": rag_chain,
        "rewrite": rewrite_chain,
    }


# ========= 工具函式 =========
def extract_clean_query(text: str) -> str:
    """從 rewriter 輸出中抽出『真正要拿去當 query 的那句話』"""
    if not text:
        return ""

    # 1) 若有 <answer>...</answer>，優先用裡面的內容
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2) 把 <think>...</think> 整塊砍掉
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 3) 只留最後一行（通常是實際問題）
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1]


def remove_thinking_tags(text: str) -> str:
    """從回應文本中移除模型的思考過程（<think>...</think> 標籤）"""
    if not text:
        return ""
    # 移除所有 <think>...</think> 標籤及其內容
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


@traceable(name="API-Ask", run_type="chain", metadata={"app": "campus_rag_api"})
def ask(chain_dict: dict, q: str):
    """執行查詢：先用 LLM 做時間改寫，再丟給 RAG"""
    rag_chain = chain_dict["rag"]
    rewrite_chain = chain_dict["rewrite"]

    now = datetime.now(ZoneInfo("Asia/Taipei"))
    roc_year = now.year - 1911
    m, d = now.month, now.day

    # 學年學期計算：8/1 新學年；2/1 第二學期
    if (m, d) >= (8, 1):
        acad_year = roc_year
        sem = "第一學期"
    elif (m, d) >= (2, 1):
        acad_year = roc_year - 1
        sem = "第二學期"
    else:
        acad_year = roc_year - 1
        sem = "第一學期"

    acad_term = f"{acad_year}學年{sem}"
    now_str = f"民國{roc_year}年{m}月{d}日 {now.strftime('%H:%M')}"

    # 先讓 rewrite_chain 把相對時間改寫
    try:
        rewritten_q = rewrite_chain.invoke({
            "query": q,
            "now": now_str,
        }).strip()
        rewritten_q = extract_clean_query(rewritten_q)
    except Exception:
        rewritten_q = q

    if not rewritten_q:
        rewritten_q = q

    print(f"[DEBUG] rewritten query: {rewritten_q!r}")

    # 再把改寫後的 query 丟進 RAG
    result = rag_chain.invoke({
        "input": rewritten_q,
        "now": now_str,
        "acad_term": acad_term,
        "original_query": q,
        "rewritten_query": rewritten_q,
    })

    # ⭐⭐ 在這裡把 <think>...</think> 移掉，只留下真正要顯示的回答
    raw_answer = result.get("answer", "")
    clean_answer = remove_thinking_tags(raw_answer)
    result["answer"] = clean_answer

    return result


# ========= FastAPI App & Lifespan =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用啟動 / 關閉生命週期"""
    global reranker, chains

    print("🔄 載入 reranker 模型...")
    reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")
    print("🔄 建立 RAG chains...")
    chains = build_chain()
    print("✅ 模型載入完成！")

    yield

    print("👋 關閉應用...")


app = FastAPI(title="TTU CSE Chatbot API", lifespan=lifespan)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= 基本路由 =========
@app.get("/")
async def root():
    return {"message": "TTU CSE Chatbot API is running"}


@app.get("/favicon.ico")
async def favicon():
    """防止 favicon 404 錯誤"""
    return {"message": "No favicon"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_ready": chains is not None,
    }


# ========= SSE 串流 =========
async def generate_stream(message: str) -> AsyncGenerator[str, None]:
    """生成 SSE 串流回應（使用新的 RAG + rewriter 邏輯）"""
    try:
        global chains
        if chains is None:
            raise RuntimeError("RAG chains 尚未初始化")

        # 在背景 thread 跑完整 ask（含 rewrite + RAG）
        result = await asyncio.to_thread(ask, chains, message)
        response_text = result.get("answer", "")

        # 逐字串流
        for char in response_text:
            payload = {"content": char, "done": False}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)

        # 發送完成訊號
        yield f"data: {json.dumps({'content': '', 'done': True}, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_msg = f"抱歉，處理您的問題時發生錯誤：{str(e)}"
        payload = {"content": error_msg, "done": True, "error": True}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/chat/stream")
async def chat_stream(message: str):
    """SSE 串流聊天端點（GET /api/chat/stream?message=...）"""
    return StreamingResponse(
        generate_stream(message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ========= 非串流 API =========
@app.post("/api/chat")
async def chat(request: dict):
    """非串流聊天端點（使用新的 RAG + rewriter 邏輯）"""
    try:
        global chains
        if chains is None:
            raise RuntimeError("RAG chains 尚未初始化")

        message = request.get("message", "")

        result = await asyncio.to_thread(ask, chains, message)

        return {
            "response": result.get("answer", ""),
            "original_query": result.get("original_query", message),
            "rewritten_query": result.get("rewritten_query", message),
            "now": result.get("now"),
            "acad_term": result.get("acad_term"),
            "sources": [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance": doc.metadata.get("relevance"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                }
                for doc in result.get("context", [])
            ],
        }
    except Exception as e:
        return {
            "response": f"抱歉，處理您的問題時發生錯誤：{str(e)}",
            "error": True,
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
