# ingest.py
import os
import math
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any

# 強制使用 safetensors 格式以避免 PyTorch 2.5.1 的安全限制
os.environ["TRANSFORMERS_PREFER_SAFETENSORS"] = "1"
os.environ["SENTENCE_TRANSFORMERS_USE_SAFETENSORS"] = "1"

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

DATA_DIR = Path("data")
DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"

# =========================
# JSON schema 自動偵測
# =========================
def detect_schema(obj: Any) -> str:
    """
    回傳 "people" / "news" / "unknown"
    - people: 有「人物」「電話」「信箱」等鍵
    - news:   有 "url","title","published_at","content"
    """
    sample = None
    if isinstance(obj, list) and obj:
        sample = obj[0]
    elif isinstance(obj, dict):
        sample = obj
    else:
        return "unknown"

    keys = set(sample.keys())
    if {"人物", "電話", "信箱"} & keys:
        return "people"
    if {"url", "title", "published_at", "content"} <= keys:
        return "news"
    return "unknown"


# =========================
# people（老師名錄） adapter
# =========================
_name_title_pat = re.compile(r"^\s*(?P<name>[\u4e00-\u9fa5A-Za-z0-9．・\s]+)\s*(?P<title>.+)?$")

def _parse_name_title(s: str) -> Dict[str, str]:
    m = _name_title_pat.match(s or "")
    if not m:
        return {"name": s or "", "title": ""}
    name = (m.group("name") or "").strip().replace("\u00a0", " ")
    title = (m.group("title") or "").strip(" ,，")
    return {"name": name, "title": title}

def _fmt_people_page_content(meta: Dict[str, Any]) -> str:
    return "\n".join([
        f"姓名：{meta.get('name','')}",
        f"職稱/職務：{meta.get('title','')}",
        f"辦公室：{meta.get('office','')}",
        f"分機/電話：{meta.get('phone','')}",
        f"Email：{meta.get('email','')}",
        f"研究領域：{meta.get('expertise','')}",
    ])

def people_records_to_documents(data: List[Dict[str, Any]], source_path: str) -> List[Document]:
    docs: List[Document] = []
    for i, rec in enumerate(data, 1):
        who = _parse_name_title(rec.get("人物", ""))
        meta = {
            "source": source_path,
            "file_type": "json",
            "content_type": "people",
            "name": who["name"],
            "title": who["title"],
            "phone": rec.get("電話"),
            "email": rec.get("信箱"),
            "office": rec.get("辦公室"),
            # 原始 JSON 多半把研究領域塞在 "metadata" 欄位前綴字串；這裡清掉前綴
            "expertise": (rec.get("metadata") or "").replace("教學與研究領域 :", "").strip(),
            "idx": i,
            # 標示：這類文件不需要再 split
            "needs_split": False,
        }
        docs.append(Document(page_content=_fmt_people_page_content(meta), metadata=meta))
    return docs  # 一人一段，不再切塊


# =========================
# news（系網新聞） adapter
# =========================
def _fmt_news_page_content(meta: Dict[str, Any], content: str) -> str:
    return "\n".join([
        f"標題：{meta.get('title','')}",
        f"日期：{meta.get('published_at','')}",
        "內文：",
        content or "",
    ])

def news_records_to_documents(data: List[Dict[str, Any]], source_path: str) -> List[Document]:
    docs: List[Document] = []

    # 目標：控制每塊長度，避免嵌入模型截斷（中文字數≈token 數量的好近似）
    TARGET_CHARS = 1000      # 大致對應 256–384 tokens
    OVERLAP_CHARS = 80

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TARGET_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        # 強化中文標點分割；最後再用空白、英文標點補刀
        separators=["\n\n", "\n", "。", "！", "？", "；", "、", "：", "——", " ", ",", ".", "，", ":"]
    )

    from datetime import datetime
    def to_ts(s: str | None) -> int | None:
        if not s: return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
            try:
                return int(datetime.strptime(s, fmt).timestamp())
            except Exception:
                pass
        return None

    for i, rec in enumerate(data, 1):
        title = rec.get("title") or ""
        content = rec.get("content") or ""
        published_at = rec.get("published_at")
        published_ts = to_ts(published_at)
        url = rec.get("url")

        # 為每篇新聞產生穩定 article_id（利於重組與去重）
        article_key = f"{source_path}|{url or title}|{published_at or ''}|{i}"
        article_id = hashlib.sha1(article_key.encode("utf-8")).hexdigest()[:12]

        base_meta = {
            "source": source_path,
            "file_type": "json",
            "content_type": "news",
            "url": url,
            "title": title,
            "published_at": published_at,
            "published_at_ts": published_ts,  # 之後好用於排序/過濾
            "idx": i,
            "article_id": article_id,
            # 我們會在此函式完成切塊，避免主流程再次切
            "needs_split": False,
        }

        # 短文直接一塊（避免不必要切割）
        if len(content) <= TARGET_CHARS:
            docs.append(Document(
                page_content=_fmt_news_page_content(base_meta, content),
                metadata=base_meta
            ))
            continue

        # 長文：只對「內文」做切塊，再把標題/日期當前綴補回每塊
        parts = splitter.split_text(content)

        # 若最後一塊太短，併回前一塊，避免產生「碎尾」
        if len(parts) >= 2 and len(parts[-1]) < TARGET_CHARS // 3:
            parts[-2] = parts[-2] + ("\n" if not parts[-2].endswith("\n") else "") + parts[-1]
            parts.pop()

        for j, part in enumerate(parts):
            meta = dict(base_meta)
            meta.update({"chunk": j})
            docs.append(Document(
                page_content=_fmt_news_page_content(meta, part),
                metadata=meta
            ))

    return docs



# =========================
# JSON 載入與分派
# =========================
def load_json_as_documents(path: Path) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    schema = detect_schema(obj)
    if schema == "people":
        data = obj if isinstance(obj, list) else [obj]
        return people_records_to_documents(data, str(path))
    elif schema == "news":
        data = obj if isinstance(obj, list) else [obj]
        return news_records_to_documents(data, str(path))
    else:
        # 後備：不認得的 JSON → 扁平化成一份 Document（仍保留 metadata）
        def flatten(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield str(k); yield from flatten(v)
            elif isinstance(o, list):
                for it in o:
                    yield from flatten(it)
            else:
                yield str(o)
        text = "\n".join(x for x in flatten(obj) if x)
        return [Document(page_content=text, metadata={"source": str(path), "type": "unknown", "needs_split": True})]


# =========================
# 其他格式載入
# =========================
def load_documents(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in data_dir.rglob("*"):
        suf = path.suffix.lower()
        if suf == ".pdf":
            loader = PyPDFLoader(str(path))
            # PDF loader 已經一頁一份，後面仍建議切一下長頁（交由主流程）
            for i, d in enumerate(loader.load(), 1):
                d.metadata.update({"source": str(path), "type": "pdf", "needs_split": True, "idx": i})
                docs.append(d)
        elif suf == ".docx":
            loader = Docx2txtLoader(str(path))
            for i, d in enumerate(loader.load(), 1):
                d.metadata.update({"source": str(path), "type": "docx", "needs_split": True, "idx": i})
                docs.append(d)
        elif suf == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": str(path), "type": "txt", "needs_split": True,"idx": 1}))
        elif suf == ".csv":
            loader = CSVLoader(file_path=str(path), encoding="utf-8")
            for i, d in enumerate(loader.load(), 1):
                d.metadata.update({"source": str(path), "type": "csv", "needs_split": True, "idx": i})
                docs.append(d)
        elif suf in {".json", ".jsonl"}:
            # 你的兩個 JSON（department_members.json, ttu_cse_news.sorted.json）會走這條
            docs.extend(load_json_as_documents(path))
        else:
            # 忽略其他格式
            continue
    return docs


# =========================
# 主流程（建立/更新索引）
# =========================
def main():
    assert DATA_DIR.exists(), "請先把檔案放進 data/ 目錄（支援 JSON/PDF/DOCX/CSV/TXT）"

    print("▶ 讀取檔案…")
    docs = load_documents(DATA_DIR)
    print(f"▶ 讀到 {len(docs)} 份原始文件/片段（含已切好的 JSON 文件）")

    # 將需要再切塊的文件挑出來切，其他直接保留
    need_split: List[Document] = [d for d in docs if d.metadata.get("needs_split", False)]
    keep: List[Document] = [d for d in docs if not d.metadata.get("needs_split", False)]

    if need_split:
        print(f"▶ 對 {len(need_split)} 份文件進一步切塊…")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        
        split_more: List[Document] = []
        for doc in need_split:
            parts = splitter.split_documents([doc])  # 對單一文件切，保持分組
            for j, part in enumerate(parts):
                part.metadata["needs_split"] = False
                part.metadata["chunk"] = j          # 每份原始文件的塊序
            split_more.extend(parts)

        final_docs = keep + split_more
        print(f"▶ 產生 {len(split_more)} 個切塊；合計 {len(final_docs)} 份可入庫文件")
    else:
        final_docs = keep
        print(f"▶ 無需額外切塊；合計 {len(final_docs)} 份可入庫文件")

    # 統計各 type 方便你檢查
    from collections import Counter
    type_count = Counter(d.metadata.get("type", "unknown") for d in final_docs)
    print("▶ 類型統計：", dict(type_count))

    print("▶ 準備嵌入模型(多語)…")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
    )

    def logistic_from_distance(d: float) -> float:
        return 1.0 / (1.0 + math.exp(d))

    print("▶ 建立/更新 Chroma 向量庫…")
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"},   # 距離度量：cosine / l2 / ip
        relevance_score_fn=logistic_from_distance       # 距離→0~1 的轉換
    )

    # 穩定 ID：避免重複寫入；同內容+同來源會得到相同 id
    def stable_id(doc: Document) -> str:
        src = str(doc.metadata.get("source", ""))
        typ = str(doc.metadata.get("type", ""))
        file_type = str(doc.metadata.get("file_type", ""))
        content_type = str(doc.metadata.get("content_type", doc.metadata.get("type", "")))  # 與舊 'type' 相容
        aid = str(doc.metadata.get("article_id", ""))  # people/其他型別沒有就留空
        idx = str(doc.metadata.get("idx", ""))
        chk = str(doc.metadata.get("chunk", 0))
        raw = f"{src}|{file_type}|{content_type}|{aid}|{idx}|{chk}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    ids = [stable_id(d) for d in final_docs]
    # 除錯
    from collections import Counter
    dups = [k for k, v in Counter(ids).items() if v > 1]
    if dups:
        raise RuntimeError(f"stable_id 重覆 {len(dups)} 筆，例：{dups[:3]}")

    vectordb.add_documents(final_docs, ids=ids)

    print("✅ 完成索引建立/更新！資料庫位置：", DB_DIR)
    print(f"✅ collection = {COLL_NAME}，共新增/去重後寫入 {len(final_docs)} 份文件")


if __name__ == "__main__":
    main()
