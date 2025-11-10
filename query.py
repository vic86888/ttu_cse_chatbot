# query.py
from typing import Set
import os

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

DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"

def make_scored_retriever(vdb, k: int = 10):
    def _retrieve(query: str):
        def as_docs(pairs):
            out = []
            for doc, score in pairs:
                md = dict(doc.metadata) if doc.metadata else {}
                md["relevance"] = float(score)
                out.append(Document(page_content=doc.page_content, metadata=md))
            return out

        # 判斷是否像在問「新聞/最新消息」
        q = (query or "").lower()
        prefer_news = any(kw in q for kw in ["新聞", "消息", "news", "最新"])

        docs: list[Document] = []
        if prefer_news:
            # LangChain Chroma 支援 filter=dict → 對應 Chroma where
            news_pairs = vdb.similarity_search_with_relevance_scores(
                query, k=k, filter={"content_type": "news"}
            )
            docs.extend(as_docs(news_pairs))

            if len(docs) < k:
                more_pairs = vdb.similarity_search_with_relevance_scores(query, k=k)
                docs.extend(as_docs(more_pairs))

            # 去重（source+article_id 或 source+idx）
            seen = set()
            uniq = []
            for d in docs:
                md = d.metadata or {}
                key = ("article", md.get("source"), md.get("article_id")) if md.get("article_id") \
                    else ("row", md.get("source"), md.get("idx"))
                if key in seen: 
                    continue
                seen.add(key)
                uniq.append(d)
            docs = uniq[:k]
        else:
            pairs = vdb.similarity_search_with_relevance_scores(query, k=k)
            docs = as_docs(pairs)

        return docs

    return RunnableLambda(_retrieve).with_config({
        "run_name": "ChromaRetriever(scored)",
        "tags": ["retriever", "chroma", "with-scores"],
        "metadata": {"k": k}
    })

def build_chain():
    # 1) LLM
    llm = ChatOllama(
        model="cwchang/llama-3-taiwan-8b-instruct:latest",
        temperature=0,
    ).with_config({
        "run_name": "Ollama-LLM",
        "tags": ["ollama", "tw-8b", "local"],
        "metadata": {"provider": "ollama"},
    })

    # 2) 提示詞
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是大同大學資工系問答機器人。你會得到跟問題相關的文件，你只依據提供的文件內容回答問題，"
         "若無法從文件中找到答案，請清楚說明。請以繁體中文作答。\n\n"
         "{context}"),
        ("human", "{input}")
    ]).with_config({
        "run_name": "StuffPrompt",
        "tags": ["prompt", "stuff"],
    })

    # 3) stuff chain
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt).with_config({
        "run_name": "StuffDocumentsChain",
        "tags": ["chain", "stuff"],
    })

    # 4) 向量庫 & 檢索器（含分數）
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda"},
    )
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    # ✨ 關鍵：建立 scored_retriever，然後用 itemgetter 抽出 input 字串再餵檢索器
    scored_retriever = make_scored_retriever(vectordb, k=20)
    retriever_runnable = itemgetter("input") | scored_retriever  # dict -> str -> [Document]

    # 5) RAG 鏈（retriever + combine_docs）
    rag_chain = create_retrieval_chain(retriever_runnable, doc_chain).with_config({
        "run_name": "CampusRAG",
        "tags": ["campus-rag", "cli"],
    })
    return rag_chain

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

        raw = md.get("relevance")
        try:
            score = float(raw) if raw is not None else None
        except Exception:
            score = None
        score_str = f"{score:.3f}" if score is not None else "—"

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
        rows.append(f"{header}\n   └ 分數：{score_str}｜片段：{snippet}")

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
            res = ask(chain, q)
            print("\n🧠 答案：\n", res["answer"], "\n", sep="")
            print("📚 來源、分數與片段：")
            print(pretty_print_snippets_with_scores(res["context"]))
            print("-" * 60)
    except KeyboardInterrupt:
        print("\n再見！")
