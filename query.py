# query.py
from typing import Set
import os
import re

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

# Rich 套件用於美化終端輸出
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

Prompt.prompt_suffix = ""  # 或 " " 之類的，避免預設的冒號

console = Console()

DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"

# 你原本的關鍵字
EVENT_KEYWORDS = ["新聞","消息","news","最新","最近","活動","說明會","講座","論壇","營隊","徵才","行事曆"]

# 時間窗參數（自行調整）
NEWS_LOOKBACK_DAYS = 120   # 新聞：只抓過去 N 天（含今天）
CAL_PAST_DAYS      = 60    # 行事曆：抓過去 N 天
CAL_FUTURE_DAYS    = 180   # 行事曆：抓未來 N 天

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
    k_retrieve = max(k * 4, 100)

    def _retrieve(query: str):
        def as_docs(pairs):
            out = []
            for doc, score in pairs:
                md = dict(doc.metadata) if doc.metadata else {}
                md["relevance"] = float(score)
                out.append(Document(page_content=doc.page_content, metadata=md))
            return out

        q_lower = (query or "").lower()
        prefer_news = any(kw in q_lower for kw in EVENT_KEYWORDS)

        docs: list[Document] = []

        if prefer_news:
            now_ts = int(datetime.now(ZoneInfo("Asia/Taipei")).timestamp())

            # --- 1) 新聞：只抓過去 NEWS_LOOKBACK_DAYS 天 ---
            news_cutoff = now_ts - NEWS_LOOKBACK_DAYS * 24 * 3600
            news_pairs = vdb.similarity_search_with_relevance_scores(
                query,
                k=k_retrieve,
                filter={
                    "$and": [
                        {"content_type": "news"},
                        {"published_at_ts": {"$gte": news_cutoff}}
                    ]
                }
            )
            docs.extend(as_docs(news_pairs))

            # --- 2) 行事曆：抓過去 CAL_PAST_DAYS 天 + 未來 CAL_FUTURE_DAYS 天 ---
            cal_start = now_ts - CAL_PAST_DAYS * 24 * 3600
            cal_end   = now_ts + CAL_FUTURE_DAYS * 24 * 3600

            cal_pairs = vdb.similarity_search_with_relevance_scores(
                query,
                k=k_retrieve,
                filter={
                    "$and": [
                        {"content_type": "calendar"},          # 你新增的 event docs
                        {"event_date_ts": {"$gte": cal_start}},
                        {"event_date_ts": {"$lte": cal_end}},
                    ]
                }
            )
            docs.extend(as_docs(cal_pairs))

            # （可選）Fallback：如果你還沒加 calendar_events_to_documents，
            # 只會有 calendar_month chunk，這裡補抓一點避免完全空
            if not cal_pairs:
                cal_month_pairs = vdb.similarity_search_with_relevance_scores(
                    query,
                    k=min(20, k_retrieve),
                    filter={"content_type": "calendar_month"}
                )
                docs.extend(as_docs(cal_month_pairs))

            # --- 3) 不夠再補一般候選 ---
            if len(docs) < k_retrieve:
                more_pairs = vdb.similarity_search_with_relevance_scores(
                    query, k=k_retrieve
                )
                docs.extend(as_docs(more_pairs))

            # --- 4) 去重 ---
            seen = set()
            uniq = []
            for d in docs:
                md = d.metadata or {}
                key = (
                    ("page", md.get("source"), md.get("page"))
                    if md.get("page") is not None
                    else ("article", md.get("source"), md.get("article_id"))
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

        # --- 5) cross-encoder rerank（語意）---
        docs = rerank_docs(query, docs, top_n=k)

        # --- 6) prefer_news 時做「時間導向 final sort」---
        if prefer_news:
            now_ts = int(datetime.now(ZoneInfo("Asia/Taipei")).timestamp())

            def time_key(d):
                md = d.metadata or {}
                ts = md.get("published_at_ts") or md.get("event_date_ts") or 0
                rr = md.get("rerank_score") or 0.0

                # 讓「離現在最近」的排前面，且未來活動優先於過去活動
                future_flag = 0 if ts >= now_ts else 1
                delta = abs(int(ts) - now_ts)
                return (future_flag, delta, -float(rr))

            docs = sorted(docs, key=time_key)

        return docs

    return RunnableLambda(_retrieve).with_config({
        "run_name": "ChromaRetriever+Reranker",
        "tags": ["retriever", "chroma", "with-scores", "rerank"],
        "metadata": {"k": k}
    })

RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")  # 或 "cpu"

def build_chain():
    # 1) LLM
    llm = ChatOllama(
        # model="cwchang/llama-3-taiwan-8b-instruct:latest",
        model="qwen3:latest",
        temperature=0,
    ).with_config({
        "run_name": "Ollama-LLM",
        "tags": ["ollama", "tw-8b", "local"],
        "metadata": {"provider": "ollama"},
    })

    # 2) 提示詞
    from langchain_core.prompts import ChatPromptTemplate

    # 在 build_chain 函式內修改 prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "現在時間：{now}\n\n"
        "目前學期：{acad_term}\n\n"
        "你是大同大學資工系問答機器人。請根據系統提供的資訊（如現在時間、目前學期）以及與問題相關的文件內容回答問題。"
        "回答結尾需附上參考網址，若沒有則附上來源文件"
        "若無法從系統提供的資訊（如現在時間、目前學期）以及文件中找到答案，請清楚說明。請以繁體中文作答。\n\n"
        "{context}"
        ),
        ("human", "{input}")
    ]).with_config({
        "tags": ["chain", "stuff"],
    })

    # 3) stuff chain
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt).with_config({
        "run_name": "StuffDocumentsChain",
        "tags": ["chain", "stuff"],
    })

    # 4) 向量庫 & 檢索器(含分數)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        # model_kwargs={"device": "cpu"},
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
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    roc_year = now.year - 1911

    m, d = now.month, now.day

    # 依規定：8/1 開始新學年；2/1 開始第二學期
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

    # ✅ 多傳 acad_term 給 prompt
    return chain.invoke({
        "input": q,
        "now": now_str,
        "acad_term": acad_term
    })

if __name__ == "__main__":
    # 需要：export LANGSMITH_TRACING=true 與 LANGSMITH_API_KEY
    chain = build_chain()
    
    # 使用 rich 顯示歡迎訊息
    console.print(Panel.fit(
        "💬 大同大學資工系問答機器人\n輸入問題開始對話，按 Ctrl+C 結束",
        title="歡迎",
        border_style="cyan"
    ))
    
    try:
        while True:
            # 使用 rich 的 Prompt 取代 input
            console.print("[bold cyan]❓ 你的問題[/bold cyan]")
            q = Prompt.ask("")  # 空提示，讓使用者從空欄輸入
            
            if not q.strip():
                continue
            
            # 執行查詢
            res = ask(chain, q)            
            raw = res["answer"]
            # --- 修改開始：使用 Regex 解析 XML ---
            thinking = ""
            answer = raw

            # 1. 嘗試提取 <think> 區塊
            think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()

            # 2. 嘗試提取 <answer> 區塊
            answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # 如果找不到 <answer> 標籤，可能模型沒跟隨格式
                # 為了保險，如果找到了 <think>，就把剩下的當作 answer
                # 或者直接顯示原始文字
                if think_match:
                    # 把 raw 中的 <think>...</think> 移除，剩下的當作回答
                    answer = raw.replace(think_match.group(0), "").strip()
            # --- 修改結束 ---

            # 印思考
            if thinking:
                console.print("\n[bold purple]🔍 思考過程：[/bold purple]")
                console.print(Panel(
                Markdown(thinking),
                border_style="purple",
                padding=(1,2)
            ))
            
            # 印最終回答 (如果有解析失敗，answer 會是原始全文，至少不會報錯)
            console.print("\n[bold green]✅ 最終回答：[/bold green]")
            console.print(Panel(
                Markdown(answer),
                border_style="green",
                padding=(1,2)
            ))
            
            # 顯示來源資訊
            console.print("\n[bold yellow]📚 參考來源：[/bold yellow]")
            sources_text = pretty_print_snippets_with_scores(res["context"])
            console.print(sources_text)
            
            console.print("[dim]" + "─" * 80 + "[/dim]\n")
            
    except KeyboardInterrupt:
        console.print("\n[bold blue]👋 再見！[/bold blue]")
