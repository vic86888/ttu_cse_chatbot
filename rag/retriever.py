# rag/retriever.py
"""
RAG 檢索器模組
負責從向量資料庫中檢索相關文件
"""
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# 常數設定
DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# 事件關鍵字
EVENT_KEYWORDS = [
    "新聞", "消息", "news", "最新",
    "最近", "活動", "說明會", "講座", "論壇", "營隊", "徵才"
]

# 全域變數（初始化時設置）
vectordb = None
reranker = None


def initialize_retriever():
    """初始化向量資料庫和重排序模型"""
    global vectordb, reranker
    
    if vectordb is None:
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
    
    if reranker is None:
        reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")


def rerank_docs(query: str, docs: List[Document], top_n: int) -> List[Document]:
    """用 cross-encoder 對候選文件重新排序，只保留前 top_n"""
    if not docs:
        return []

    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    scored = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    out: List[Document] = []
    for doc, s in scored[:top_n]:
        md = dict(doc.metadata) if doc.metadata else {}
        md["rerank_score"] = float(s)
        out.append(Document(page_content=doc.page_content, metadata=md))

    return out


def retrieve_docs(query: str, k: int = 10) -> Tuple[List[Document], str]:
    """
    檢索與查詢相關的文件
    
    Args:
        query: 使用者查詢
        k: 返回的文件數量
        
    Returns:
        tuple: (文件列表, 來源資訊字串)
    """
    # 確保已初始化
    initialize_retriever()
    
    k_retrieve = max(k * 4, 20)
    
    def as_docs(pairs):
        out = []
        for doc, score in pairs:
            md = dict(doc.metadata) if doc.metadata else {}
            md["relevance"] = float(score)
            out.append(Document(page_content=doc.page_content, metadata=md))
        return out
    
    q = (query or "").lower()
    prefer_news = any(kw in q for kw in EVENT_KEYWORDS)
    
    docs: List[Document] = []
    if prefer_news:
        news_pairs = vectordb.similarity_search_with_relevance_scores(
            query, k=k_retrieve, filter={"content_type": "news"}
        )
        docs.extend(as_docs(news_pairs))
        
        if len(docs) < k_retrieve:
            more_pairs = vectordb.similarity_search_with_relevance_scores(
                query, k=k_retrieve
            )
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
        pairs = vectordb.similarity_search_with_relevance_scores(
            query, k=k_retrieve
        )
        docs = as_docs(pairs)
    
    # 重新排序
    docs = rerank_docs(query, docs, top_n=k)
    
    # 生成來源資訊
    sources_info = format_sources(docs)
    
    return docs, sources_info


def format_sources(docs: List[Document], max_chars: int = 200) -> str:
    """
    格式化文件來源資訊為 Markdown 格式
    
    Args:
        docs: 文件列表
        max_chars: 每個片段的最大字元數
        
    Returns:
        格式化的來源資訊字串
    """
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
    
    for d in docs:
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
        rr = md.get("rerank_score")
        rel_str = f"{float(rel):.3f}" if rel is not None else "—"
        rr_str = f"{float(rr):.3f}" if rr is not None else "—"
        
        text = (d.page_content or "").replace("\n", " ").strip()
        snippet = (text[:max_chars] + "…") if len(text) > max_chars else text
        
        extra = ""
        if page is not None:
            extra = f"（第 {page} 頁）"
        elif title:
            extra = f"（{title}）"
        elif chunk is not None:
            extra = f"（chunk {chunk}）"
        
        header = f"**{display_idx}. [{ctype}] {src}{extra}**"
        rows.append(
            f"{header}\n"
            f"- 向量分數：`{rel_str}` | Rerank 分數：`{rr_str}`\n"
            f"- 片段：{snippet}\n"
        )
    
    return "\n".join(rows) if rows else "無參考來源"
