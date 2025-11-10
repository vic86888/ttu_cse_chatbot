# ingest.py
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader  # PDF / DOCX
from langchain_community.document_loaders import PyMuPDFLoader  # ← 新增
from langchain_text_splitters import RecursiveCharacterTextSplitter            # 文本切塊
from langchain_huggingface import HuggingFaceEmbeddings                        # 嵌入
from langchain_chroma import Chroma                                            # 向量庫

# 在 ingest.py 內替換 add_documents 區塊
import hashlib
from langchain.schema import Document

DATA_DIR = Path("data")
DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"

def load_documents(data_dir: Path) -> List:
    docs = []
    for path in data_dir.rglob("*"):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))        # 每頁一個 Document，含 page 中繼資料
            docs.extend(loader.load())
        elif path.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(path))     # 讀取 .docx
            docs.extend(loader.load())
    return docs

def main():
    assert DATA_DIR.exists(), "請先把 PDF / DOCX 放進 data/ 目錄"

    print("▶ 讀取檔案…")
    docs = load_documents(DATA_DIR)

    print(f"▶ 讀到 {len(docs)} 份原始文件，開始文本切塊…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200
    )
    splits = splitter.split_documents(docs)
    print(f"▶ 產生 {len(splits)} 個切塊")

    print("▶ 準備嵌入模型（HuggingFace Sentence-Transformers）…")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # 若有安裝 GPU 版 PyTorch，打開下一行即可使用 CUDA
        model_kwargs={"device": "cuda"},
    )

    print("▶ 建立/更新 Chroma 向量庫…")
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    def stable_id(doc: Document) -> str:
        src = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", ""))
        raw  = f"{src}|{page}|{doc.page_content}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()  # 40 字元，穩定

    ids = [stable_id(d) for d in splits]

    vectordb.add_documents(splits, ids=ids)

    print("✅ 完成索引建立/更新！資料庫位置：", DB_DIR)

if __name__ == "__main__":
    main()
