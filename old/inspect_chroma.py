# inspect_chroma.py
import chromadb
from pprint import pprint

PERSIST_DIR = "storage/chroma"     # 你的 Chroma 路徑
COLL_NAME   = "campus_rag"         # 你的 collection 名稱

client = chromadb.PersistentClient(path=PERSIST_DIR)

print("=== 所有 collections ===")
# 0.6+ 版會回傳「名稱列表」而不是物件
print(client.list_collections())

print("\n=== 連線到目標 collection ===")
col = client.get_collection(COLL_NAME)
print(col)

print("\n=== 筆數（count） ===")
print(col.count())                 # collection 中的紀錄數

# # 放在 inspect_chroma.py 的最後面或呼叫處

# def print_batch(batch, start_idx=0, preview=180):
#     """把一個 batch 逐筆列印：id / source / page / 長度 / 內容前 N 字"""
#     ids   = batch.get("ids", [])
#     docs  = batch.get("documents", [])
#     metas = batch.get("metadatas", [])
#     for i, (id_, doc, md) in enumerate(zip(ids, docs, metas), start=start_idx):
#         md = md or {}
#         src  = md.get("source", "—")
#         page = md.get("page_label", md.get("page", "—"))
#         text = (doc or "").replace("\n", " ")
#         nchar = len(text)
#         print(f"#{i:05d}  id={id_}")
#         print(f"   source={src}  page={page}  chars={nchar}")
#         if text:
#             print(f"   {text[:preview]}{'…' if nchar > preview else ''}")
#         print("-" * 80)

# # 建議：不要把 embeddings 一起拿，輸出會很肥（除非你真的要看）
# BATCH_SIZE = 10
# total = col.count()  # ← 官方推薦用 count 知道總筆數
# print(f"\n=== 全庫逐批列出（共 {total} 筆；每批 {BATCH_SIZE}） ===")

# for offset in range(0, total, BATCH_SIZE):
#     batch = col.get(
#         include=["documents", "metadatas"],  # IDs 會自動回傳，不用放在 include
#         limit=BATCH_SIZE,
#         offset=offset                        # 依官方支援的分頁參數
#     )
#     print_batch(batch, start_idx=offset)

# 指定你想看的來源檔名（完全比對）
print("\n=== 指定你想看的來源檔名（完全比對) ===")
SRC = "data/大學部修業規定.pdf"

pdf_all = col.get(
    where={"source": {"$eq": SRC}},            # metadata 等值篩選
    include=["documents", "metadatas"],
    limit=10_000                                # 夠大即可，或改用分頁 while 取完
)

pprint({k: pdf_all[k] for k in ["ids", "metadatas", "documents"]})

# 依 metadata.page 排序後輸出前幾段
rows = sorted(zip(pdf_all["documents"], pdf_all["metadatas"]),
              key=lambda x: x[1].get("page", 0))
print(f"找到 {len(rows)} 個切塊，前兩段內容：\n")
for i, (doc, md) in enumerate(rows[:2], 1):
    print(f"[p{i}] {doc[:1000]}\n")
    # print(f"[p{md.get('page_label', md.get('page', 0)+1)}] {doc[:1000]}\n")
