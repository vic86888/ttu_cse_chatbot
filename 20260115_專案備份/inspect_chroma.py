# inspect_chroma.py
import chromadb
import sys
import os

# ==========================================
# 新增設定：輸出檔案功能
# ==========================================
OUTPUT_FILE = "inspection_result.txt"  # 你想存檔的檔名

class DualWriter:
    """
    這個類別會把輸出的文字同時送到：
    1. 終端機 (Terminal)
    2. 指定的文字檔 (File)
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # Python 3 需要這個方法來相容緩衝區刷新
        self.terminal.flush()
        self.logfile.flush()

# 將標準輸出重導向到我們的 DualWriter
print(f"★ 輸出日誌將同步儲存至: {OUTPUT_FILE}")
sys.stdout = DualWriter(OUTPUT_FILE)

# ==========================================
# 原本的程式邏輯
# ==========================================

PERSIST_DIR = "storage/chroma"     # 你的 Chroma 路徑
COLL_NAME   = "campus_rag"         # 你的 collection 名稱

client = chromadb.PersistentClient(path=PERSIST_DIR)

print("=== 所有 collections ===")
# 0.6+ 版會回傳「名稱列表」而不是物件
try:
    print(client.list_collections())
except Exception as e:
    print(f"列出 collection 失敗 (可能是版本差異): {e}")

print("\n=== 連線到目標 collection ===")
col = client.get_collection(COLL_NAME)
print(col)

print("\n=== 筆數（count） ===")
count = col.count()
print(count)                 # collection 中的紀錄數


# inspect_chroma.py（查詢預覽部分）

print("\n=== 指定你想看的來源檔名（不直接取全文） ===")
SRC = "data_qwen/ttu_cse_news.sorted.json"   # 你要看的來源檔
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
K = 200          # 想要預覽幾筆
PREVIEW = 500   # 每筆預覽字數
pick_ids = [rid for rid, _ in rows[:K]]

if pick_ids:
    subset = col.get(
        ids=pick_ids,
        include=["documents", "metadatas"]   # 這裡再抓少量全文
    )
    print(f"\n=== 前 {len(pick_ids)} 筆預覽（每段前 {PREVIEW} 字）===\n")

    LABEL_W = 10  # 欄位名稱的寬度
    def print_kv(label, value):
        if value is None or value == "":
            return
        print(f"{label:<{LABEL_W}}: {value}")

    # 為了讓輸出順序跟 sorted 後的一致，我們手動建立一個對照表
    # 因為 col.get(ids=...) 回傳的順序不一定等於輸入的 ids 順序
    doc_map = {rid: doc for rid, doc in zip(subset["ids"], subset["documents"])}
    meta_map = {rid: md for rid, md in zip(subset["ids"], subset["metadatas"])}

    for i, rid in enumerate(pick_ids, 1):
        doc = doc_map.get(rid)
        md = meta_map.get(rid)

        md = md or {}
        txt   = (doc or "").replace("\n", " ").strip()
        ctype = md.get("content_type", md.get("type", "unknown"))
        ftype = md.get("file_type", "—")
        src   = md.get("source", "—")
        src_url = md.get("source_url", "—")  # 額外顯示學校網址（有就印）
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
        print_kv("SourceURL", src_url)
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

print(f"\n★ 檢查完畢，結果已存入: {OUTPUT_FILE}")