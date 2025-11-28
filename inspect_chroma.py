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

# inspect_chroma.py（替換底部查詢那段）

print("\n=== 指定你想看的來源檔名（不直接取全文） ===")
SRC = "data_qwen/course_history_113.json"   # 你要看的來源檔
FILTER_CONTENT_TYPE = None              # 例如 "news"；不過濾請設 None

# 1) 先拿 ids + metadatas（不抓 documents）
where = {"source": {"$eq": SRC}}
if FILTER_CONTENT_TYPE:
    where = {"$and": [where, {"content_type": {"$eq": FILTER_CONTENT_TYPE}}]}

res = col.get(
    where=where,
    include=["metadatas"],
    limit=10_000
)

ids = res.get("ids", [])
metas = res.get("metadatas", [])
print(f"符合來源「{SRC}」的切塊數：{len(ids)}")

# 顯示該來源的 content_type 分布，方便檢查
from collections import Counter
print("content_type 分布：", dict(Counter((md or {}).get("content_type", "unknown") for md in metas)))

# 2) 依 metadata 做更合理排序：
#    - 有 page（PDF）：依 page, chunk
#    - news：依 published_at_ts 由新到舊，再依 article_id, chunk
#    - CSV/一般列：依 idx, chunk
#    - 其他：最後
def sort_key(item):
    _id, md = item
    md = md or {}
    page = md.get("page")
    if page is not None:
        return (0, int(page), int(md.get("chunk", 0)))
    if md.get("content_type") == "news":
        ts = int(md.get("published_at_ts") or 0)
        return (1, -ts, str(md.get("article_id", "")), int(md.get("chunk", 0)))
    if md.get("idx") is not None:
        return (2, int(md.get("idx")), int(md.get("chunk", 0)))
    return (3, 0, int(md.get("chunk", 0)))

rows = sorted(zip(ids, metas), key=sort_key)

# 3) 只挑前 K 筆再去取 documents 預覽
K = 40           # 想要預覽幾筆
PREVIEW = 500   # 每筆預覽字數
pick_ids = [rid for rid, _ in rows[:K]]

if pick_ids:
    subset = col.get(
        ids=pick_ids,
        include=["documents", "metadatas"]   # 這裡再抓少量全文
    )
    print(f"\n=== 前 {len(pick_ids)} 筆預覽（每段前 {PREVIEW} 字）===\n")

    LABEL_W = 10  # 欄位名稱的寬度（可自行調整 8~14 之間看起來最整齊）
    def print_kv(label, value):
        if value is None or value == "":
            return
        print(f"{label:<{LABEL_W}}: {value}")

    for i, (rid, doc, md) in enumerate(zip(subset["ids"], subset["documents"], subset["metadatas"]), 1):
        md = md or {}
        txt   = (doc or "").replace("\n", " ").strip()
        ctype = md.get("content_type", md.get("type", "unknown"))
        ftype = md.get("file_type", "—")
        src   = md.get("source", "—")
        page  = md.get("page")
        idx   = md.get("idx")
        chunk = md.get("chunk")
        title = md.get("title")
        pub   = md.get("published_at")

        # 預覽（單行截斷）
        preview = txt[:PREVIEW] + ("…" if len(txt) > PREVIEW else "")

        print(f"#{i:02d}")
        print_kv("ID", rid)
        print_kv("Type", f"{ctype}|{ftype}")
        print_kv("Source", src)
        print_kv("Title", title)
        print_kv("Date", pub)
        print_kv("Page", page)
        print_kv("Idx", idx)
        print_kv("Chunk", chunk)
        print_kv("Chars", len(txt))
        print_kv("Preview", preview)
        print("-" * 80)
else:
    print("沒有符合條件的切塊。")
