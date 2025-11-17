# query.py
from typing import Set
import os

from datetime import datetime
from zoneinfo import ZoneInfo  # 新增這行

from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langsmith import traceable
from sentence_transformers import CrossEncoder

DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"

EVENT_KEYWORDS = [
    "新聞", "消息", "news", "最新",
    "最近", "活動", "說明會", "講座", "論壇", "營隊", "徵才"
]

from langchain_core.documents import Document

def rerank_docs(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """用 cross-encoder 對候選文件重新排序，只保留前 top_n。"""
    if not docs:
        return []

    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)  # 長度 = len(docs) 的 numpy array

    scored = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    out: list[Document] = []
    for doc, s in scored[:top_n]:
        md = dict(doc.metadata) if doc.metadata else {}
        md["rerank_score"] = float(s)  # 之後 debug 好看一點
        out.append(Document(page_content=doc.page_content, metadata=md))

    return out

def make_scored_retriever(vdb, k: int = 10):
    # 先抓比較多，再給 reranker 挑前 k
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
                more_pairs = vdb.similarity_search_with_relevance_scores(
                    query, k=k_retrieve
                )
                docs.extend(as_docs(more_pairs))

            # 去重（source+article_id 或 source+idx）
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
            pairs = vdb.similarity_search_with_relevance_scores(
                query, k=k_retrieve
            )
            docs = as_docs(pairs)

        # ⭐ 最關鍵：用 cross-encoder 重新排序，只保留前 k 個
        docs = rerank_docs(query, docs, top_n=k)
        return docs

    return RunnableLambda(_retrieve).with_config({
        "run_name": "ChromaRetriever+Reranker",
        "tags": ["retriever", "chroma", "with-scores", "rerank"],
        "metadata": {"k": k}
    })

RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")  # 沒 GPU 就拿掉 device

def build_chain():
    # 1) LLM
    llm = ChatOllama(
#        model="cwchang/llama-3-taiwan-8b-instruct:latest",
        model="qwen3:latest",
        temperature=0,
    ).with_config({
        "run_name": "Ollama-LLM",
        "tags": ["ollama", "tw-8b", "local"],
        "metadata": {"provider": "ollama"},
    })

    # 2) 提示詞
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "你是大同大學資工系問答機器人。\n"
        "今天日期是：{today}，現在時間是：{now_time}（台北時間）。\n"
        "學年等於民國紀年，114學年就是2025年"
        "你會得到跟問題相關的文件，你只依據提供的文件內容回答問題，"
        "若無法從文件中找到答案，請清楚說明。請以繁體中文作答。\n\n"
        "{context}"),
        ("human", "{input}")
    ])

    # 3) stuff chain
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt).with_config({
        "run_name": "StuffDocumentsChain",
        "tags": ["chain", "stuff"],
    })

    # 4) 向量庫 & 檢索器（含分數）
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # 🔴 很推薦加
    )
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    # ✨ 關鍵：建立 scored_retriever，然後用 itemgetter 抽出 input 字串再餵檢索器
    scored_retriever = make_scored_retriever(vectordb, k=10) #　k=5　課本內容太多
    retriever_runnable = itemgetter("input") | scored_retriever  # dict -> str -> [Document]

    # 5) RAG 鏈（retriever + combine_docs）
    rag_chain = create_retrieval_chain(retriever_runnable, doc_chain).with_config({
        "run_name": "CampusRAG",
        "tags": ["campus-rag", "cli"],
    })

    # ➕ 包一層：自動加上 today
    def inject_today(inputs: dict) -> dict:
        """在每次呼叫時，動態注入今天日期字串。"""
        # 明確使用台北時間，而不是系統預設時區
        now = datetime.now(ZoneInfo("Asia/Taipei"))
        today_str = now.strftime("%Y-%m-%d")          # 例如：2025-11-17
        # 如果你想要民國格式，可以再多一個：
        roc_year = now.year - 1911
        today_roc = f"{roc_year}年{now.month}月{now.day}日"

        # 可以選擇用哪一個給 LLM，看你偏好：
        # HH:MM:SS（24 小時制）
        now_time = now.strftime("%H:%M:%S")  # 例如 "14:03:27"

        return {
            **inputs,
            "today": today_roc,
            "now_time": now_time,
        }
        # return {**inputs, "today": today_str} # 今天日期是：(民國)114年11月17日

    full_chain = RunnableLambda(inject_today) | rag_chain

    return full_chain

def pretty_print_snippets_with_scores(context_docs, max_chars: int = 240):
    seen = set()
    rows = []

    def dedup_key(d):
        md = d.metadata or {}
        src = md.get("source", "unknown")
        if md.get("page") is not None:
            return ("page", src, md.get("page"))
        if md.get("article_id") is not None:
            return ("article", src, md.get("article_id"))
        if md.get("idx") is not None:
            return ("row", src, md.get("idx"))
        return ("src", src)

    for d in context_docs:
        key = dedup_key(d)
        if key in seen:
            continue
        seen.add(key)

        md = d.metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page")
        title = md.get("title")
        chunk = md.get("chunk")
        ctype = md.get("content_type", md.get("type", "unknown"))

        display_idx = len(rows) + 1

        rel = md.get("relevance")
        rr  = md.get("rerank_score")
        rel_str = f"{float(rel):.3f}" if rel is not None else "—"
        rr_str  = f"{float(rr):.3f}" if rr is not None else "—"

        text = (d.page_content or "").replace("\n", " ").strip()
        snippet = (text[:max_chars] + "…") if len(text) > max_chars else text

        extra = ""
        if page is not None:
            extra = f"（第 {page} 頁）"
        elif title:
            extra = f"（{title}）"
        elif chunk is not None:
            extra = f"（chunk {chunk}）"

        header = f"{display_idx}. [{ctype}] {src}{extra}"
        rows.append(
            f"{header}\n"
            f"   └ 向量分數：{rel_str}｜rerank 分數：{rr_str}｜片段：{snippet}"
        )

    return "\n".join(rows)

@traceable(name="CLI-Ask", run_type="chain", metadata={"app": "campus_rag_cli"})
def ask(chain, q: str):
    return chain.invoke({"input": q})

if __name__ == "__main__":
    # 需要：export LANGSMITH_TRACING=true 與 LANGSMITH_API_KEY
    chain = build_chain()
    print("💬 請輸入你的問題（Ctrl+C 結束）：")
    try:
        while True:
            q = input("> ")
            # q += time()
            res = ask(chain, q)
            print("\n🧠 答案：\n", res["answer"], "\n", sep="")
            print("📚 來源、分數與片段：")
            print(pretty_print_snippets_with_scores(res["context"]))
            print("-" * 60)
    except KeyboardInterrupt:
        print("\n再見！")
