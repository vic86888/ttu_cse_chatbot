#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chroma Cleaner (互動版)
- 輸入檔名：刪除該來源檔在向量庫中的所有切塊（比對 metadata.source）
- 輸入以 / 結尾的路徑（例如 data/）：刪除該資料夾底下所有來源
- 輸入 ALL：清空整個 collection（刪除所有切塊）

需求：
    pip install chromadb
"""

import sys
from pathlib import Path
from typing import List, Set

import chromadb

# 與 ingest.py 保持一致
PERSIST_DIR = "storage/chroma"
COLL_NAME   = "campus_rag"

BATCH_SIZE_ALL  = 10_000   # 清空時每批刪除筆數
BATCH_SCAN      = 10_000   # 掃描 metadatas 時每批讀取筆數


def connect_collection(persist_dir: str, coll_name: str):
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        col = client.get_collection(coll_name)
    except Exception as e:
        names = client.list_collections()
        print(f"[錯誤] 找不到 collection：{coll_name}")
        print(f"       目前可用 collections：{names}")
        sys.exit(1)
    return client, col


def count_where(col, where: dict | None = None, batch: int = BATCH_SCAN) -> int:
    """以 where 條件計數（以分批 col.get 疊加）。"""
    total = 0
    offset = 0
    where = where or {}
    while True:
        res = col.get(where=where, include=[], limit=batch, offset=offset)
        n = len(res.get("ids", []))
        total += n
        if n < batch:
            break
        offset += batch
    return total


def list_unique_sources(col) -> List[str]:
    """掃描整個 collection，回傳出現過的唯一 source 值列表。"""
    sources: Set[str] = set()
    offset = 0
    while True:
        res = col.get(include=["metadatas"], limit=BATCH_SCAN, offset=offset)
        metas = res.get("metadatas", [])
        if not metas:
            break
        for md in metas:
            src = (md or {}).get("source")
            if src:
                sources.add(str(src))
        if len(metas) < BATCH_SCAN:
            break
        offset += BATCH_SCAN
    return sorted(sources)


def match_sources(candidates: List[str], user_input: str) -> List[str]:
    """
    以幾種規則尋找相符來源：
      1) 完整相等
      2) basename 相等（Path(...).name）
      3) 路徑以 user_input 作結尾（方便輸入相對路徑或檔名）
    """
    name = user_input.strip()
    matches_exact = [s for s in candidates if s == name]
    if matches_exact:
        return matches_exact

    matches_basename = [s for s in candidates if Path(s).name == name]
    if len(matches_basename) == 1:
        return matches_basename

    matches_suffix = [s for s in candidates if s.replace("\\", "/").endswith(name.replace("\\", "/"))]
    return matches_basename if matches_basename else matches_suffix


def delete_all(col) -> None:
    """清空整個 collection（逐批刪除，保留 collection 殼）。"""
    before = col.count()
    print(f"\n[清空全部] 目前筆數：{before}")
    removed = 0
    while True:
        batch = col.get(include=[], limit=BATCH_SIZE_ALL, offset=0)
        ids = batch.get("ids", [])
        if not ids:
            break
        col.delete(ids=ids)
        removed += len(ids)
        remain = col.count()
        print(f"  - 刪除 {len(ids)} 筆，剩餘 {remain}")
        if remain == 0:
            break
    after = col.count()
    print(f"[完成] 原本 {before} → 現在 {after}（共刪除 {removed} 筆）")


def delete_by_source(col, src: str) -> None:
    """刪除指定來源檔（metadata.source == src）的所有切塊。"""
    print(f"\n[刪除指定來源] source = {src}")
    where = {"source": {"$eq": src}}
    matched = count_where(col, where=where)
    if matched == 0:
        print("  - 找不到符合切塊，未執行刪除。")
        return
    before = col.count()
    col.delete(where=where)
    after = col.count()
    print(f"[完成] collection 由 {before} → {after}（預期刪除 {matched} 筆）")


def delete_by_source_prefix(col, prefix: str) -> None:
    """
    刪除所有 metadata.source 以 prefix 開頭的來源，例如 prefix='data/'。
    不動其他來源（例如 data_qwen/）。
    """
    print(f"\n[刪除資料夾] prefix = {prefix}")
    sources = list_unique_sources(col)
    norm_prefix = prefix.replace("\\", "/")
    targets = [s for s in sources if s.replace("\\", "/").startswith(norm_prefix)]

    if not targets:
        print("  - 找不到任何符合該 prefix 的來源檔，未執行刪除。")
        return

    print(f"  - 將刪除 {len(targets)} 個來源檔：")
    for s in targets:
        print(f"    - {s}")

    confirm = input("⚠️ 確定要刪除以上所有來源？(yes/no) ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("已取消刪除。")
        return

    total_before = col.count()
    deleted_expected = 0

    for src in targets:
        where = {"source": {"$eq": src}}
        n = count_where(col, where=where)
        if n == 0:
            continue
        deleted_expected += n
        col.delete(where=where)
        print(f"  - 已刪除來源 {src} 的 {n} 筆切塊")

    total_after = col.count()
    print(f"[完成] collection 由 {total_before} → {total_after}（預期刪除約 {deleted_expected} 筆）")


def main():
    print(f"連線到 Chroma：{PERSIST_DIR} / collection={COLL_NAME}")
    _, col = connect_collection(PERSIST_DIR, COLL_NAME)
    total = col.count()
    print(f"目前切塊總數：{total}")

    try:
        user = input(
            "\n請輸入要刪除的檔名或路徑：\n"
            " - 輸入檔名（或完整路徑）：只刪除該檔案來源\n"
            " - 輸入以 '/' 結尾的路徑（例如 data/）：刪除此資料夾底下所有來源\n"
            " - 輸入 ALL：清空整個 collection\n"
            " - 直接 Enter 離開\n\n> "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已離開。")
        return

    if not user:
        print("未輸入，離開。")
        return

    if user.upper() == "ALL":
        confirm = input("⚠️ 確定要清空整個 collection？(yes/no) ").strip().lower()
        if confirm in {"y", "yes"}:
            delete_all(col)
        else:
            print("已取消清空。")
        return

    # 🔹 新增：以 prefix 刪除整個資料夾（例如 data/）
    if user.endswith("/") or user.endswith("\\"):
        delete_by_source_prefix(col, user)
        return

    # 原本邏輯：刪除單一來源檔
    print("\n搜尋來源中，請稍候…")
    sources = list_unique_sources(col)
    matches = match_sources(sources, user)

    if not matches:
        print("找不到符合的來源檔，未進行任何動作。")
        return

    if len(matches) == 1:
        delete_by_source(col, matches[0])
        return

    # 多筆符合：讓使用者選擇
    print("找到多個可能的來源：")
    for i, s in enumerate(matches, 1):
        print(f"  [{i}] {s}")
    pick = input("請輸入要刪除的編號（Enter 取消）： ").strip()
    if not pick:
        print("已取消。")
        return
    if not pick.isdigit() or not (1 <= int(pick) <= len(matches)):
        print("輸入無效，未進行任何動作。")
        return
    delete_by_source(col, matches[int(pick) - 1])


if __name__ == "__main__":
    main()
