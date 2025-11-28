# ingest.py
import os
import json
import re
import hashlib
import sys
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

from datetime import datetime
from zoneinfo import ZoneInfo

# ingest.py 開頭
from json_rewriter import rewrite_json_record

DATA_DIR = Path("data_qwen")
DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"

# =========================
# JSON schema 自動偵測
# =========================
def detect_schema(obj: Any) -> str:
    """
    回傳 "people" / "news" / "school" / "unknown"
    - people: 有「人物」「電話」「信箱」等鍵
    - news:   有 "url","title","published_at","content"（可額外含 "category"）
    - school: 有「名稱」「英文名稱」「校訓」等鍵（例如 about_schoo.json）
    """
    sample = None
    if isinstance(obj, list) and obj:
        sample = obj[0]
    elif isinstance(obj, dict):
        # 檢查是否是新格式的教職員資料（有 "總覽" 和 "成員列表" 鍵）
        if "總覽" in obj and isinstance(obj.get("總覽"), dict):
            overview = obj["總覽"]
            if "成員列表" in overview:
                return "people"
        # 檢查是否是舊格式的教職員資料（有 "成員列表" 鍵）
        if "成員列表" in obj and isinstance(obj["成員列表"], list):
            return "people"
        elif isinstance(obj, dict):
            # --- ✅ 新格式：課程歷史（巢狀：113上/113下 → 年級 → 課程列表） ---
            term_keys = [
                k for k in obj.keys()
                if re.match(r"^\d{2,3}[上下]$", str(k).strip())
            ]
            if term_keys:
                v0 = obj.get(term_keys[0])
                if isinstance(v0, dict):
                    # 任一 grade block 內含「課程列表」就視為新格式
                    if any(isinstance(gv, dict) and "課程列表" in gv for gv in v0.values()):
                        return "course_history_nested"
        sample = obj
    else:
        return "unknown"

    keys = set(sample.keys())
    # 老師名錄 (支援兩種格式: 舊格式用「人物」, 新格式用「姓名」+「職稱」+「系所」)
    if {"人物", "電話", "信箱"} & keys or {"姓名", "職稱", "信箱"} <= keys:
        return "people"
    # 系網新聞
    if {"url", "title", "published_at", "content"} <= keys:
        return "news"
    # 學校基本資料（about_schoo.json）
    if {"名稱", "英文名稱", "校訓"} <= keys:
        return "school"
    if "類別" in keys and ("內容" in keys or "說明" in keys):
        return "academic_rules"
    if ("辦理項目" in keys and "承辦人" in keys) or ("學系" in keys and "聯絡人員" in keys):
        return "contacts"
    if {"學年學期", "課號", "課程名稱", "教師"} <= keys:
        return "course_history"
    if {"選別", "學年學期", "所屬年級", "課程名稱"} <= keys and not ({"課號", "教師"} & keys):
        return "course_overview"
    # 人工智慧學分學程（或其他學分學程）課程清單
    # 特色鍵：有「設置宗旨/適用對象/課程代碼/課程名稱/學分數」
    if {"設置宗旨", "適用對象", "課程代碼", "課程名稱", "學分數"} <= keys:
        return "program_courses"
        # 行事曆 / 校務日程
    # 特色鍵：有「年/月/日/活動事項」（通常還有 星期、資料來源）
    if {"年", "月", "日", "活動事項"} <= keys:
        return "calendar"
    return "unknown"

# =========================
# 新格式 course_history（巢狀：學期→年級→課程列表） adapter
#  - overview 的 idx 使用全域遞增 int（避免 stable_id 重複）
#  - overview 內容含「教師 / 選別 / 學分」
#  - overview 依 token ≤ 500 動態分批，且不拆單一課程條目
#  - 每個 overview chunk 最後附資料來源 URL
#  - term/grade 先排序，確保 idx 穩定
# =========================

def course_history_nested_to_documents(
    obj: Dict[str, Any], source_path: str
) -> List[Document]:
    docs: List[Document] = []

    overview_global_idx = 0   # overview 全域 idx（int, 檔內唯一）
    global_course_idx = 0     # 每門課全域 idx（目前未用，先保留以後可能會用）

    def as_int(x, default=None):
        try:
            s = str(x).strip()
            if not s:
                return default
            return int(s)
        except Exception:
            return default

    def parse_year_term(s: str) -> tuple[int | None, str]:
        """把 '113上' / '113下' 拆成 (113, '上'/'下')"""
        s = (s or "").strip()
        if not s:
            return None, ""
        for i, ch in enumerate(s):
            if not ch.isdigit():
                year = as_int(s[:i], None)
                term = s[i:]
                return year, term
        return as_int(s, None), ""

    def term_sort_key(t: str):
        """學期排序：年小→大；同年 上→下"""
        year, term = parse_year_term(t)
        term_order = 0 if term == "上" else 1 if term == "下" else 9
        return (year or 0, term_order, t)

    GRADE_ORDER = {
        "一年級": 1, "二年級": 2, "三年級": 3, "四年級": 4, "研究所": 10,
    }

    def grade_sort_key(g: str):
        return (GRADE_ORDER.get(g, 99), g)

    def parse_credits(x: Any) -> float | None:
        try:
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None

    # ===== token 計數與不拆課程的 batching =====
    def count_chars(text: str) -> int:
        # 直接用 Python 字串長度（以 Unicode 字元計）
        return len(text or "")

    def batch_courses_by_chars(
        header_lines: List[str],
        course_entries: List[Dict[str, Any]],
        tail_lines: List[str],
        max_chars: int = 500,
    ) -> List[List[Dict[str, Any]]]:
        """
        course_entries 每一筆是一門課（不可拆）。
        依字元數上限分批（含 header+tail），回傳「課程 entry 的列表列表」，
        每一個 batch 之後會丟給重寫器。
        """
        batches: List[List[Dict[str, Any]]] = []

        fixed_text = "\n".join(header_lines + tail_lines)
        fixed_chars = count_chars(fixed_text)

        # header + tail 已經超過上限，就全部塞同一批（交給 LLM 自己控制字數）
        if fixed_chars >= max_chars:
            batches.append(course_entries)
            return batches

        cur_chars = fixed_chars
        current_batch: List[Dict[str, Any]] = []

        for entry in course_entries:
            line = f"- {entry.get('概要', '')}"
            line_chars = count_chars(line)

            if current_batch and (cur_chars + line_chars) > max_chars:
                batches.append(current_batch)
                current_batch = []
                cur_chars = fixed_chars

            # 單一課程自己就超過 max_chars：仍要放（不拆課）
            current_batch.append(entry)
            cur_chars += line_chars

        if current_batch:
            batches.append(current_batch)

        return batches

    # ===== term 先排序（113上 → 113下）=====
    for year_term in sorted(obj.keys(), key=term_sort_key):
        grades_block = obj.get(year_term)
        if not isinstance(grades_block, dict):
            continue

        year, term = parse_year_term(str(year_term))

        # ===== grade 先排序（一 → 二 → 三 → 四）=====
        for grade_name in sorted(grades_block.keys(), key=grade_sort_key):
            grade_data = grades_block.get(grade_name)
            if not isinstance(grade_data, dict):
                continue

            course_list = grade_data.get("課程列表", []) or []
            if not isinstance(course_list, list):
                course_list = [course_list]

            course_count = grade_data.get("課程數")
            try:
                course_count = int(course_count)
            except Exception:
                course_count = len(course_list)

            # ========= 預先整理 overview 的課程 entry（不可拆原子） =========
            course_entries: List[Dict[str, Any]] = []
            data_sources_all: List[str] = []

            for c in course_list:
                if not isinstance(c, dict):
                    continue

                name     = str(c.get("課程名稱", "")).strip()
                code     = str(c.get("課號", "")).strip()
                teacher  = str(c.get("教師", "")).strip()
                category = str(c.get("選別", "")).strip()
                credits  = parse_credits(c.get("學分"))
                ds       = str(c.get("資料來源", "")).strip()

                if ds:
                    data_sources_all.append(ds)

                if not name:
                    continue

                summary = f"{name}"
                if code:
                    summary += f"({code})"
                if teacher:
                    summary += f" / {teacher}"
                if category:
                    summary += f" / {category}"
                if credits is not None:
                    summary += f" / {credits}學分"

                entry: Dict[str, Any] = {
                    "課程名稱": name,
                    "課號": code,
                    "教師": teacher,
                    "選別": category,
                    "學分": credits,
                    "資料來源": ds,
                    "概要": summary,
                }
                course_entries.append(entry)

            if not course_entries:
                continue

            data_source_str = "；".join(sorted(set(data_sources_all)))

            # ========== (A) overview docs（≤500 字元，不拆課） ==========
            header_lines = [
                f"學年學期：{year_term}",
                f"所屬年級：{grade_name}",
                f"課程數：{course_count}",
                "",
                "課程名單：",
            ]
            tail_lines: List[str] = []
            if data_source_str:
                tail_lines = ["", f"資料來源：{data_source_str}"]

            # 依字數拆成多個「課程總覽 chunk」
            batches = batch_courses_by_chars(
                header_lines=header_lines,
                course_entries=course_entries,
                tail_lines=tail_lines,
                max_chars=500,
            )

            num_chunks = len(batches)

            for chunk_idx, course_batch in enumerate(batches):
                overview_global_idx += 1

                # 準備給重寫器的 record：一個 chunk = 某學期某年級的一部分課程總覽
                record: Dict[str, Any] = {
                    "學年學期": str(year_term),
                    "年級": str(grade_name),
                    "課程總數": course_count,
                    "本批課程數": len(course_batch),
                    "課程列表": course_batch,          # list[dict]，每筆是一門課
                    "資料來源": data_source_str,
                    "來源檔案": source_path,
                }

                if overview_global_idx == 9:  # 只印前兩個 chunk，避免爆 log
                    print("[DEBUG course_history record]")
                    print(json.dumps(record, ensure_ascii=False, indent=2))
                    # 然後再呼叫 rewrite_json_record(...)

                try:
                    overview_text = rewrite_json_record(
                        record=record,
                        schema_hint="course_history_overview",
                        max_chars=500,
                    )
                except Exception as e:
                    print(f"[course_history_nested_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
                    sys.exit(1)

                # metadata：這個 chunk 內的課程概要字串（原本叫 course_names）
                course_summaries_in_chunk = [
                    entry.get("概要", "") for entry in course_batch if entry.get("概要")
                ]

                docs.append(Document(
                    page_content=overview_text.strip(),
                    metadata={
                        "source": source_path,
                        "file_type": "json",
                        "type": "course_history_overview",
                        "content_type": "course_history_overview",

                        "year_term": str(year_term),
                        "year": year,
                        "term": term,
                        "grade": str(grade_name),

                        "course_count": course_count,
                        "course_names": "、".join(course_summaries_in_chunk),
                        "data_source": data_source_str,

                        "idx": overview_global_idx,  # int
                        "chunk": chunk_idx,
                        "total_chunks": num_chunks,
                        "needs_split": False,
                    }
                ))

    return docs


# =========================
# calendar.json（行事曆：依月分切塊） adapter
# =========================

def calendar_months_to_documents(
    data: List[Dict[str, Any]], source_path: str
) -> List[Document]:
    docs: List[Document] = []
    if not data:
        return docs

    def to_int(x) -> int | None:
        try:
            s = str(x).strip()
            return int(s) if s else None
        except Exception:
            return None

    def parse_day_start(day_raw: str) -> int | None:
        """
        抓「起始日」排序用：
        - "1" -> 1
        - "8~12" -> 8
        - "10/13~11/3" -> 13 (取起始日)
        解析失敗就 None
        """
        s = (day_raw or "").strip()
        if not s:
            return None
        # 取第一段可能的數字
        m = re.search(r"(\d+)", s)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    # 1) 依 (年, 月) 分組
    grouped: Dict[tuple[int | None, int | None], List[Dict[str, Any]]] = {}
    for rec in data:
        y = to_int(rec.get("年"))
        m = to_int(rec.get("月"))
        grouped.setdefault((y, m), []).append(rec)

    # 2) 依年/月排序輸出
    month_items = sorted(grouped.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0))

    idx = 0
    for (year_roc, month), items in month_items:
        idx += 1
        year_ad = year_roc + 1911 if year_roc is not None else None

        # 先按「起始日」粗排序（None 的保持原順序）
        items_sorted = sorted(
            items,
            key=lambda r: (
                parse_day_start(str(r.get("日", ""))) is None,
                parse_day_start(str(r.get("日", ""))) or 0,
            ),
        )

        # 3) 整理成活動列表（給重寫器 & metadata 用）
        events_entries: List[Dict[str, Any]] = []
        events_for_meta: List[str] = []
        data_sources: List[str] = []

        for r in items_sorted:
            day_raw = str(r.get("日", "")).strip()
            weekday = str(r.get("星期", "")).strip()
            event = str(r.get("活動事項", "")).strip()
            ds = str(r.get("資料來源", "")).strip()

            if ds:
                data_sources.append(ds)

            # 給 metadata 用的簡單字串
            if event:
                events_for_meta.append(f"{month}/{day_raw}:{event}")

            # 給重寫器用的結構化活動資訊
            events_entries.append(
                {
                    "日": day_raw,
                    "星期": weekday,
                    "活動事項": event,
                    "資料來源": ds,
                }
            )

        data_source_str = "；".join(sorted(set(data_sources)))

        # 4) 準備這個「月份總覽」的 record，丟給重寫器
        header_title = f"{year_roc if year_roc is not None else ''}年{month if month is not None else ''}月行事曆"

        record: Dict[str, Any] = {
            "行事曆標題": header_title,
            "民國年": year_roc,
            "西元年": year_ad,
            "月份": month,
            "活動數量": len(events_entries),
            "活動列表": events_entries,
            "資料來源": data_source_str,
            "來源檔案": source_path,
        }

        try:
            text = rewrite_json_record(
                record=record,
                schema_hint="calendar_month",  # 對應這種「月份總覽」資料
                max_chars=500,
            )
        except Exception as e:
            print(f"[calendar_months_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
            sys.exit(1)

        # 5) 特化 metadata（維持原本欄位）
        meta = {
            "source": source_path,
            "file_type": "json",
            "type": "calendar_month",
            "content_type": "calendar_month",

            "title": str(items_sorted[0].get("title", "")).strip(),
            "year_roc": year_roc,
            "year_ad": year_ad,
            "month": month,

            "event_count": len(items_sorted),
            "events": "、".join(events_for_meta),   # 存成字串，方便 filter / 檢索
            "data_source": data_source_str,

            "idx": idx,
            "needs_split": False,  # 月 chunk 不再二次切
        }

        docs.append(Document(page_content=text.strip(), metadata=meta))

    return docs

def calendar_events_to_documents(
    data: List[Dict[str, Any]], source_path: str
) -> List[Document]:
    """
    將行事曆每一筆活動獨立成一份 Document，
    並補 event_date/event_date_ts 讓 retriever 能用時間 filter。
    內容本體改用 rewrite_json_record 做自然語句重寫。
    """
    docs: List[Document] = []
    if not data:
        return docs

    tz = ZoneInfo("Asia/Taipei")

    def to_int(x) -> int | None:
        try:
            s = str(x).strip()
            return int(s) if s else None
        except Exception:
            return None

    def parse_range_start(day_raw: str) -> tuple[int | None, int | None]:
        """
        從「日」欄抓起始(月,日)：
        - "1" -> (None, 1)
        - "8~12" -> (None, 8)
        - "10/13~11/3" -> (10, 13)
        """
        s = (day_raw or "").strip()
        if not s:
            return None, None

        m = re.match(r"(\d+)\s*/\s*(\d+)\s*~\s*(\d+)\s*/\s*(\d+)", s)
        if m:
            return int(m.group(1)), int(m.group(2))

        m = re.match(r"(\d+)\s*~\s*(\d+)", s)
        if m:
            return None, int(m.group(1))

        m = re.search(r"(\d+)", s)
        if m:
            return None, int(m.group(1))

        return None, None

    idx = 0
    for rec in data:
        year_roc = to_int(rec.get("年"))
        month = to_int(rec.get("月"))
        day_raw = str(rec.get("日", "")).strip()

        start_m_raw, start_d = parse_range_start(day_raw)
        start_m = start_m_raw if start_m_raw is not None else month

        year_ad = year_roc + 1911 if year_roc is not None else None

        event_date_iso = None
        event_date_ts = None
        if year_ad and start_m and start_d:
            event_date_iso = f"{year_ad:04d}-{start_m:02d}-{start_d:02d}"
            event_date_ts = int(datetime(year_ad, start_m, start_d, tzinfo=tz).timestamp())

        weekday = str(rec.get("星期", "")).strip()
        activity = str(rec.get("活動事項", "")).strip()
        url = str(rec.get("資料來源", "")).strip()
        title = str(rec.get("title", "行事曆")).strip()

        idx += 1

        # === 準備給重寫器的 record ===
        record: Dict[str, Any] = {
            "標題": title,
            "民國年": year_roc,
            "西元年": year_ad,
            "月份": month,
            "原始日欄位": day_raw,
            "解析起始月份": start_m,
            "解析起始日": start_d,
            "星期": weekday,
            "活動事項": activity,
            "活動日期": event_date_iso,
            "活動日期_timestamp": event_date_ts,
            "資料來源": url,
            "來源檔案": source_path,
            "原始紀錄": rec,   # 保險：把原始 JSON 也塞進去，讓 LLM 可以看到全部欄位
        }

        try:
            text = rewrite_json_record(
                record=record,
                schema_hint="calendar_event",  # 單筆行事曆活動
                max_chars=400,
            )
        except Exception as e:
            print(f"[calendar_events_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
            sys.exit(1)

        meta = {
            "source": source_path,
            "file_type": "json",

            # ✅ 給 retriever/filter 用
            "type": "calendar_event",
            "content_type": "calendar_event",

            "title": title,
            "year_roc": year_roc,
            "year_ad": year_ad,
            "month": start_m,
            "day_raw": day_raw,
            "weekday": weekday,

            "event_date": event_date_iso,
            "event_date_ts": event_date_ts,   # ✅ 關鍵：epoch int

            "activity": activity,
            "url": url,
            "idx": idx,
            "needs_split": False,
        }

        docs.append(Document(page_content=text.strip(), metadata=meta))

    return docs
    
# =========================
# program_courses.json（以課程類別分組切塊） adapter
# =========================

def program_courses_to_documents(
    data: List[Dict[str, Any]], source_path: str
) -> List[Document]:
    docs: List[Document] = []

    if not data:
        return []

    # 取學程層級資訊（每筆都一樣，拿第一筆即可）
    program_title = str(data[0].get("title", "")).strip()
    program_purpose = str(data[0].get("設置宗旨", "")).strip()
    program_target = str(data[0].get("適用對象", "")).strip()

    def parse_credits(x: Any) -> float | None:
        try:
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None

    def extract_substitutes(note: str) -> str:
        note = note or ""
        alts = re.findall(r"【([^】]+)】", note)
        alts = [a.strip() for a in alts if a.strip()]
        return "、".join(alts)

    # 1) 依課程類別分組
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in data:
        cat = str(rec.get("課程類別", "")).strip() or "未分類"
        grouped.setdefault(cat, []).append(rec)

    # 2) 每個類別 → 一份 Document（丟給重寫器）
    for idx, (cat, items) in enumerate(grouped.items(), 1):
        course_names: List[str] = []
        required_count = 0
        credits_sum = 0.0

        # 準備給重寫器用的「課程列表」
        course_entries: List[Dict[str, Any]] = []

        for rec in items:
            code = str(rec.get("課程代碼", "")).strip()
            name = str(rec.get("課程名稱", "")).strip()
            credits = parse_credits(rec.get("學分數"))
            note = str(rec.get("備註", "")).strip()
            note = "" if note.lower() == "nan" else note

            required = ("必修" in note) or ("必選" in note)
            if required:
                required_count += 1

            if credits is not None:
                credits_sum += credits

            substitutes = extract_substitutes(note)

            # 給重寫器看的單筆課程資訊
            entry: Dict[str, Any] = {
                "課程代碼": code,
                "課程名稱": name,
                "學分數": credits,
                "備註": note,
                "是否必修": required,
                "可替代課程": substitutes,  # 從備註中抽出的替代課程資訊
            }
            course_entries.append(entry)

            # 給 metadata 用的簡單名稱字串
            if name and code:
                course_names.append(f"{name}({code})")
            elif name:
                course_names.append(name)

        # 準備整個「學程＋類別」的 record，丟給重寫器
        record: Dict[str, Any] = {
            "學程名稱": program_title,
            "學程設置宗旨": program_purpose,
            "學程適用對象": program_target,
            "課程類別": cat,
            "課程列表": course_entries,
            "課程總數": len(items),
            "必修課程數": required_count,
            "總學分數": credits_sum,
            "來源檔案": source_path,
        }

        try:
            text = rewrite_json_record(
                record=record,
                schema_hint="program_courses",   # 對應這種「學程課程」資料
                max_chars=500,
            )
            # 🔍 加這兩行看看實際輸出長怎樣
            if idx == 1:
                print("\n[DEBUG program_courses_to_documents] sample output:")
                print(text[:200])
        except Exception as e:
            print(f"[program_courses_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
            sys.exit(1)

        meta = {
            "source": source_path,
            "file_type": "json",
            "type": "program_course_category",
            "content_type": "program_course_category",

            # 學程層級
            "program_title": program_title,
            "program_purpose": program_purpose,
            "program_target": program_target,

            # 類別層級（切塊 key）
            "course_category": cat,
            "course_count": len(items),
            "required_count": required_count,
            "credits_sum": credits_sum,

            # 為了 metadata 可索引、不能放 list → 轉字串
            "courses": "、".join(course_names),

            "idx": idx,
            "needs_split": False,
        }

        docs.append(Document(page_content=text.strip(), metadata=meta))

    return docs

# =========================
# course_overview.json（課程總覽） adapter
# =========================

def course_overview_to_documents(
    data: List[Dict[str, Any]], source_path: str
) -> List[Document]:
    docs: List[Document] = []

    def parse_year_term(s: str) -> tuple[int | None, str]:
        """簡單把 '113上' 拆成 (113, '上')，失敗就 (None, '')。"""
        s = (s or "").strip()
        if not s:
            return None, ""
        year = None
        for i, ch in enumerate(s):
            if not ch.isdigit():
                try:
                    year = int(s[:i])
                except Exception:
                    year = None
                term = s[i:]
                return year, term
        try:
            return int(s), ""
        except Exception:
            return None, ""

    for i, rec in enumerate(data, 1):
        if not isinstance(rec, dict):
            # 保險處理：非 dict 就包成一個欄位
            rec = {"value": rec}

        year_term_raw = str(rec.get("學年學期", "")).strip()
        year, term = parse_year_term(year_term_raw)

        select_type = str(rec.get("選別", "")).strip()        # 必修 / 選修
        grade = str(rec.get("所屬年級", "")).strip()          # 一年級 / 二年級 / 三年級 / 四年級
        data_source = str(rec.get("資料來源", "")).strip()    # URL

        names = rec.get("課程名稱") or []
        if not isinstance(names, list):
            names = [names]
        # 清掉空字串，全部轉成 str
        names = [str(n).strip() for n in names if str(n).strip()]
        course_count = len(names)
        courses_str = "、".join(names)

        # 準備給重寫器的 record
        record: Dict[str, Any] = dict(rec)  # 複製一份，避免動到原資料

        # 補充結構化資訊給 LLM 參考
        record.setdefault("學年學期", year_term_raw)
        record.setdefault("解析學年度", year)           # int 或 None
        record.setdefault("解析學期", term)             # "上" / "下" / ""
        record.setdefault("所屬年級", grade)
        record.setdefault("選別", select_type)
        record.setdefault("課程名稱列表", names)
        record.setdefault("課程數", course_count)
        record.setdefault("資料來源", data_source)
        record.setdefault("來源檔案", source_path)

        try:
            rewritten = rewrite_json_record(
                record=record,
                schema_hint="course_overview",   # 對應 detect_schema 中的類型
                max_chars=400,
            )
        except Exception as e:
            print(f"[course_overview_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
            sys.exit(1)

        text = rewritten.strip()

        meta = {
            "source": source_path,
            "file_type": "json",
            "type": "course_overview",         # 給統計/除錯用
            "content_type": "course_overview", # 之後 filter 用這個

            "year_term": year_term_raw,
            "year": year,                      # int 或 None
            "term": term,                      # "上" / "下" / ""
            "grade": grade,                    # 一年級 / 二年級 / 三年級 / 四年級

            "select_type": select_type,        # 必修 / 選修
            "course_count": course_count,      # int
            "courses": courses_str,            # ✅ 字串，不是 list
            "data_source": data_source,        # 資料來源 URL

            "idx": i,
            "needs_split": False,              # 不再切塊
        }

        docs.append(Document(page_content=text, metadata=meta))

    return docs

# =========================
# course_history.json（歷年課程資料） adapter
# =========================

def course_records_to_documents(
    data: List[Dict[str, Any]], source_path: str
) -> List[Document]:
    docs: List[Document] = []

    def parse_year_term(s: str) -> tuple[int | None, str]:
        # e.g. "113上" / "113下"
        s = (s or "").strip()
        if not s:
            return None, ""
        year = None
        for i, ch in enumerate(s):
            if not ch.isdigit():
                try:
                    year = int(s[:i])
                except Exception:
                    year = None
                term = s[i:]
                return year, term
        try:
            return int(s), ""
        except Exception:
            return None, ""

    for i, rec in enumerate(data, 1):
        yt = rec.get("學年學期", "")
        year, term = parse_year_term(yt)

        code = rec.get("課號", "") or ""
        name = rec.get("課程名稱", "") or ""
        teacher = rec.get("教師", "") or ""
        category = rec.get("選別", "") or ""
        dept = rec.get("所屬系所", "") or ""
        grade = rec.get("所屬年級", "") or ""
        data_source = rec.get("資料來源", "") or ""

        credit_raw = rec.get("學分", None)
        try:
            credits = float(credit_raw) if credit_raw not in (None, "", " ") else None
        except Exception:
            credits = None

        # 給 LLM 的文字
        lines = [
            f"學年學期：{yt}",
            f"所屬年級：{grade}",
            f"課號：{code}",
            f"課程名稱：{name}",
            f"教師：{teacher}",
            f"選別：{category}",
            f"學分：{credits if credits is not None else ''}",
            f"所屬系所：{dept}",
        ]
        if data_source:
            lines.append(f"資料來源：{data_source}")
        text = "\n".join(lines)

        meta = {
            "source": source_path,
            "file_type": "json",
            "type": "course_history",      # 統計用
            "content_type": "course",      # 之後 filter 用這個

            "year_term": yt,
            "year": year,                  # int or None
            "term": term,                  # "上" / "下" / ""
            "grade": grade,                # 一年級 / 二年級 / 三年級 / 四年級

            "course_code": code,
            "course_name": name,
            "teacher": teacher,
            "category": category,          # "必修" / "選修"
            "required": (category == "必修"),
            "credits": credits,            # float or None
            "department": dept,
            "data_source": data_source,    # 資料來源 URL

            "idx": i,
            "needs_split": False,
        }

        docs.append(Document(page_content=text, metadata=meta))

    return docs

# =========================
# contact.json（聯絡資訊） adapter
# =========================

def contact_records_to_documents(
    data: List[Dict[str, Any]], source_path: str
) -> List[Document]:
    docs: List[Document] = []

    for i, rec in enumerate(data, 1):
        data_source = rec.get("資料來源", "") or ""

        # === 判斷聯絡類型（role）＆抽 metadata 用的欄位 ===
        if "辦理項目" in rec:  # 行政/招生類
            role = "service"
            item = str(rec.get("辦理項目", "")).strip()
            person = str(rec.get("承辦人", "")).strip()
            ext = str(rec.get("分機", "")).strip()
        elif "學系" in rec:   # 各學系聯絡人
            role = "department"
            item = ""  # 這類沒有「辦理項目」
            dept = str(rec.get("學系", "")).strip()
            person = str(rec.get("聯絡人員", "")).strip()
            ext = str(rec.get("分機", "")).strip()
        else:
            role = "unknown"
            item = ""
            dept = str(rec.get("學系", "")).strip() if "學系" in rec else ""
            person = str(rec.get("承辦人") or rec.get("聯絡人員") or "").strip()
            ext = str(rec.get("分機") or "").strip()

        # === 準備給重寫器的 record ===
        record: Dict[str, Any] = dict(rec)  # 拷貝一份，避免直接改到原始資料

        # 補充一些語意提示欄位，讓 LLM 好寫一點
        if role == "service":
            record.setdefault("聯絡類型", "行政或招生相關服務聯絡資訊")
        elif role == "department":
            record.setdefault("聯絡類型", "學系聯絡人資訊")
        else:
            record.setdefault("聯絡類型", "一般聯絡資訊")

        if data_source:
            record.setdefault("資料來源", data_source)
        record.setdefault("來源檔案", source_path)

        try:
            rewritten = rewrite_json_record(
                record=record,
                schema_hint="contacts",   # 對應你 detect_schema 的類型
                max_chars=400,
            )
        except Exception as e:
            print(f"[contact_records_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
            sys.exit(1)

        text = rewritten.strip()

        # === metadata 保留原本設計 ===
        meta = {
            "source": source_path,
            "file_type": "json",
            "type": "contact",           # 給你統計用
            "content_type": "contact",   # 之後 filter 用這個
            "role": role,                # "service" or "department" or "unknown"
            "item": rec.get("辦理項目") or "",
            "department": rec.get("學系") or "",
            "person": rec.get("承辦人") or rec.get("聯絡人員") or "",
            "phone": rec.get("分機") or "",
            "data_source": data_source,
            "idx": i,
            "needs_split": False,
        }

        docs.append(Document(page_content=text, metadata=meta))

    return docs

# =========================
# academic_requirements.json（學則/畢業規定） adapter
# =========================

def academic_records_to_documents(
    data: List[Dict[str, Any]], source_path: str
) -> List[Document]:
    docs: List[Document] = []

    def infer_topic(category: str) -> str:
        if "修業規定" in category:
            return "graduation"
        if "專題" in category:
            return "capstone"
        if "輔系" in category:
            return "minor"
        if "轉系" in category:
            return "transfer"
        if "實習" in category:
            return "internship"
        if "口試" in category:
            return "thesis_oral"
        return "general"

    for i, rec in enumerate(data, 1):
        category = str(rec.get("類別", "")).strip()
        topic = infer_topic(category)

        # 準備給重寫器的 record：
        # 先複製原本的 rec，並補上推論出來的 topic、來源等資訊
        record: Dict[str, Any] = dict(rec)
        record["推論主題"] = topic              # 給 LLM 一點語意提示
        record["來源檔案"] = source_path

        try:
            rewritten = rewrite_json_record(
                record=record,
                schema_hint="academic_rules",    # 學籍 / 修業規定 / 專題 / 實習 等規定
                max_chars=500,                   # 可以稍微長一點，視需要再調整
            )
        except Exception as e:
            print(f"[academic_records_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
            sys.exit(1)

        text = rewritten.strip()

        meta = {
            "source": source_path,
            "file_type": "json",
            "type": "academic_rules",
            "content_type": "academic_rule",
            "category": category,
            "topic": topic,   # 單一字串，方便 filter
            "idx": i,
            "needs_split": False,
        }

        docs.append(Document(page_content=text, metadata=meta))

    return docs

# =========================
# school（學校資訊） adapter
# =========================

def school_info_to_documents(obj: Any, source_path: str) -> List[Document]:
    """
    將 about_school.json 這類「學校資訊」整理成一份 Document，
    並在 metadata 裡補充常用欄位（校名、校訓、網址等）。
    主體內容改為交給 rewrite_json_record 做自然語句重寫。
    """
    # 預期格式：list[dict]
    if not isinstance(obj, list) or not obj:
        # 非預期格式就先走原本的簡單 fallback，不呼叫重寫器
        text = str(obj)
        meta = {
            "source": source_path,
            "file_type": "json",
            "content_type": "school",
            "needs_split": False,
            "idx": 1,
        }
        return [Document(page_content=text, metadata=meta)]

    # 做個安全的 helper：從不同 block 抓 key
    def find_key(key: str, default: str = "") -> Any:
        for block in obj:
            if isinstance(block, dict) and key in block:
                return block[key]
        return default

    # ---------- 抽出結構化欄位（作為 metadata 用） ----------
    # 1) 基本校務
    name = find_key("名稱", "")
    name_en = find_key("英文名稱", "")
    motto = find_key("校訓", "")
    founded_at = find_key("成立時間", "")
    founder = find_key("創辦人", "")
    school_type = find_key("類型", "")

    # 2) 聯絡資訊
    address = find_key("地址", "")
    phone = find_key("電話", "")
    emergency_phone = find_key("緊急校安專線", "")
    fax = find_key("傳真", "")
    president_phone = find_key("校長室電話", "")
    president_fax = find_key("校長室傳真", "")
    president_email = find_key("校長室 email", "")

    # 3) 其他校務
    school_code = find_key("學校代碼", "")
    url = find_key("網址", "")
    departments = find_key("系所結構", [])
    if isinstance(departments, list):
        departments_str = "、".join(map(str, departments))
    else:
        departments_str = str(departments) if departments else ""

    student_count = find_key("學生人數", "")
    mascots = find_key("校友吉祥物", [])
    if isinstance(mascots, list):
        mascots_str = "、".join(map(str, mascots))
    else:
        mascots_str = str(mascots) if mascots else ""

    # 4) 歷史沿革
    prev_name = find_key("前身", "")
    reorg_at = find_key("改制時間", "")
    rename_at = find_key("更名時間", "")

    # 5) 辦學特色
    feature = find_key("特色", "")
    focus_fields = find_key("重點領域", [])
    if isinstance(focus_fields, list):
        focus_fields_str = "、".join(map(str, focus_fields))
    else:
        focus_fields_str = str(focus_fields) if focus_fields else ""

    philosophy = find_key("辦學理念", "")
    alliance = find_key("聯盟", "")

    # ---------- 準備給重寫器的 record（包含原始區塊） ----------
    record: Dict[str, Any] = {
        # 基本校務
        "名稱": name,
        "英文名稱": name_en,
        "校訓": motto,
        "成立時間": founded_at,
        "創辦人": founder,
        "類型": school_type,

        # 聯絡資訊
        "地址": address,
        "電話": phone,
        "緊急校安專線": emergency_phone,
        "傳真": fax,
        "校長室電話": president_phone,
        "校長室傳真": president_fax,
        "校長室 email": president_email,

        # 其他校務
        "學校代碼": school_code,
        "網址": url,
        "系所結構": departments,      # 保留原始 list（如果有）
        "學生人數": student_count,
        "校友吉祥物": mascots,         # 保留原始 list（如果有）

        # 歷史沿革
        "前身": prev_name,
        "改制時間": reorg_at,
        "更名時間": rename_at,

        # 辦學特色
        "特色": feature,
        "重點領域": focus_fields,      # 保留原始 list（如果有）
        "辦學理念": philosophy,
        "聯盟": alliance,

        # 保險：把原始 blocks 也放進去，讓 LLM 可以看到完整 JSON
        "原始區塊列表": obj,
    }

    try:
        # max_chars 可以視情況調整，學校簡介通常可以稍微長一點
        rewritten = rewrite_json_record(
            record=record,
            schema_hint="school_info",
            max_chars=500,
        )
    except Exception as e:
        print(f"[school_info_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
        sys.exit(1)

    text = rewritten.strip()

    # ---------- metadata 保持你原本的設計 ----------
    meta = {
        "source": source_path,
        "file_type": "json",
        "content_type": "school",

        # 1) 基本校務
        "name": name,
        "name_en": name_en,
        "motto": motto,
        "founded_at": founded_at,
        "founder": founder,
        "school_type": school_type,

        # 2) 聯絡資訊
        "address": address,
        "phone": phone,
        "emergency_phone": emergency_phone,
        "fax": fax,
        "president_phone": president_phone,
        "president_fax": president_fax,
        "president_email": president_email,

        # 3) 其他校務
        "school_code": school_code,
        "url": url,
        "departments": departments_str,    # 已經是 "、" 串好的字串
        "student_count": student_count,
        "mascots": mascots_str,

        # 4) 歷史沿革
        "prev_name": prev_name,
        "reorg_at": reorg_at,
        "rename_at": rename_at,

        # 5) 辦學特色
        "feature": feature,
        "focus_fields": focus_fields_str,
        "philosophy": philosophy,
        "alliance": alliance,

        "needs_split": False,
        "idx": 1,
    }

    return [Document(page_content=text, metadata=meta)]

# =========================
# people（老師名錄） adapter
# =========================

def people_overview_to_documents(
    data: List[Dict[str, Any]],
    source_path: str,
    max_chars: int = 500,
) -> List[Document]:
    docs: List[Document] = []
    if not data:
        return docs

    # -------- helpers --------
    def count_chars(text: str) -> int:
        return len(text or "")

    def batch_lines_by_chars(
        header_lines: List[str],
        item_lines: List[str],
        tail_lines: List[str],
        max_chars: int,
    ) -> List[List[str]]:
        """
        依照字數把 header + item_lines + tail 切成多批，每批字數不超過 max_chars（盡量）。
        回傳的每個 batch 仍是「字串列表」，我們後面會再從中解析出成員清單。
        """
        batches: List[List[str]] = []

        fixed_text = "\n".join(header_lines + tail_lines)
        fixed_chars = count_chars(fixed_text)
        if fixed_chars >= max_chars:
            # header + tail 已經超過上限，就不再細切，全部塞一批
            batches.append(header_lines + item_lines + tail_lines)
            return batches

        cur_chars = fixed_chars
        cur_items: List[str] = []

        for line in item_lines:
            lc = count_chars(line)
            if cur_items and (cur_chars + lc) > max_chars:
                batches.append(header_lines + cur_items + tail_lines)
                cur_items = []
                cur_chars = fixed_chars

            cur_items.append(line)
            cur_chars += lc

        if cur_items:
            batches.append(header_lines + cur_items + tail_lines)
        return batches

    def is_faculty_title(title: str) -> bool:
        t = title or ""
        if "系務助理" in t:
            return False
        # 只要含教授系職稱就算 faculty（含兼任）
        return any(k in t for k in ["講座教授", "教授", "副教授", "助理教授"])

    def rank_group(title: str) -> str:
        t = title or ""
        if "講座教授" in t:
            return "chair_professor"
        if "兼任" in t and "教授" in t:
            return "adjunct_professor"
        # 注意判斷順序：先副教授/助理教授，再教授
        if "副教授" in t:
            return "associate_professor"
        if "助理教授" in t:
            return "assistant_professor"
        if "教授" in t:
            return "professor"
        return "other"

    # -------- collect faculty --------
    faculty_rows = []
    dept_set = set()
    ds_set = set()

    for rec in data:
        title = str(rec.get("職稱", "") or rec.get("人物", "")).strip()
        if not is_faculty_title(title):
            continue

        name = str(rec.get("姓名", "")).strip()
        if not name:
            # 舊格式 fallback
            name = str(rec.get("人物", "")).strip()

        dept = str(rec.get("系所", "")).strip()
        if dept:
            dept_set.add(dept)

        ds = str(rec.get("資料來源", "")).strip()
        if ds:
            ds_set.add(ds)

        # overview 單行（不可拆的原子），後面用來切 batch & 還原成員列表
        line = f"{name} / {title}"
        faculty_rows.append((rank_group(title), line, name))

    if not faculty_rows:
        return docs

    data_source_str = "；".join(sorted(ds_set))
    departments_str = "、".join(sorted(dept_set))

    # -------- build overview scopes --------
    overview_idx = 0

    def emit_scope(scope: str, group: str, lines: List[str]):
        """
        scope: "faculty_all" 或 "rank_group"
        group: 職級代碼（rank_group 時有值，faculty_all 時為 ""）
        lines: 例如 ["張三 / 教授", "李四 / 副教授", ...]
        """
        nonlocal overview_idx, docs

        header = "教授總覽" if scope == "faculty_all" else f"{group} 總覽"
        header_lines = [header, "成員列表："]
        item_lines = [f"- {ln}" for ln in lines]
        tail_lines = ["", f"資料來源：{data_source_str}"] if data_source_str else []

        # 先用原本的字數邏輯切成多個 batch，再對每個 batch 丟給重寫器
        batches = batch_lines_by_chars(header_lines, item_lines, tail_lines, max_chars)
        total_chunks = len(batches)

        for chunk_i, batch_lines in enumerate(batches):
            # 從 batch_lines 中解析出該 chunk 的成員清單（姓名 / 職稱）
            member_items = []
            for ln in batch_lines:
                ln = ln.strip()
                if not ln.startswith("- "):
                    continue
                raw = ln[2:].strip()  # 去掉前面的 "- "
                if " / " in raw:
                    name_part, title_part = raw.split(" / ", 1)
                else:
                    name_part, title_part = raw, ""
                member_items.append({
                    "姓名": name_part.strip(),
                    "職稱": title_part.strip(),
                })

            # 如果這個 chunk 沒有任何成員，就略過
            if not member_items:
                continue

            overview_idx += 1

            # 準備給重寫器用的 JSON record
            record = {
                "總覽標題": header,
                "範圍類型": "全部教授" if scope == "faculty_all" else "職級分組",
                "職級代碼": group if scope == "rank_group" else "",
                "系所": departments_str,
                "成員列表": member_items,
                "資料來源": data_source_str,
            }

            try:
                text = rewrite_json_record(
                    record=record,
                    schema_hint="department_members_overview",
                    max_chars=max_chars,
                )
            except Exception as e:
                print(f"[people_overview_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
                sys.exit(1)

            names_in_chunk = [m["姓名"] for m in member_items]

            docs.append(Document(
                page_content=text.strip(),
                metadata={
                    "source": source_path,
                    "file_type": "json",
                    "type": "people_overview",
                    "content_type": "people_overview",

                    "overview_scope": scope,
                    "rank_group": group if scope == "rank_group" else "",

                    "people_count": len(member_items),
                    "departments": departments_str,
                    "names": "、".join(names_in_chunk),
                    "data_source": data_source_str,

                    "idx": overview_idx,     # overview 內全域 int
                    "chunk": chunk_i,
                    "total_chunks": total_chunks,
                    "needs_split": False,
                }
            ))

    # (1) faculty_all：全體教授
    all_lines = [line for _, line, _ in faculty_rows]
    emit_scope("faculty_all", "", all_lines)

    # (2) rank_group：依職級分組
    grouped: Dict[str, List[str]] = {}
    for rg, line, _name in faculty_rows:
        grouped.setdefault(rg, []).append(line)

    # 固定輸出順序
    order = ["chair_professor", "professor", "associate_professor", "assistant_professor", "adjunct_professor"]
    for rg in order:
        lines = grouped.get(rg, [])
        if not lines:
            continue
        emit_scope("rank_group", rg, lines)

    return docs

_name_title_pat = re.compile(
    r"^\s*(?P<name>[\u4e00-\u9fa5A-Za-z0-9．・]+)\s*(?P<title>.+)?$"
)

def _parse_name_title(s: str) -> Dict[str, str]:
    m = _name_title_pat.match(s or "")
    if not m:
        return {"name": s or "", "title": ""}

    name = (m.group("name") or "").strip().replace("\u00a0", " ")
    title = (m.group("title") or "").strip(" ,，").replace("\u00a0", " ")

    # 有些來源會把「職稱/職務：」這段也塞進來，這邊順便清掉開頭的標籤
    if title.startswith("職稱/職務"):
        title = re.sub(r"^職稱/職務[:：]?\s*", "", title)

    return {"name": name, "title": title}

def people_records_to_documents(
    data: List[Dict[str, Any]], source_path: str
) -> List[Document]:
    docs: List[Document] = []
    for i, rec in enumerate(data, 1):
        # ===== 這一段是你原本抓姓名/職稱/系所/來源 =====
        if "姓名" in rec:
            who = {"name": rec.get("姓名", "").strip(), "title": rec.get("職稱", "").strip()}
        else:
            who = _parse_name_title(rec.get("人物", ""))

        dept = rec.get("系所") or rec.get("department") or "大同大學 資訊工程學系"
        src_url = rec.get("資料來源") or rec.get("source_url") or ""

        # ===== 新增：用 LLM 把這一筆 JSON 轉成敘述句 =====
        try:
            rewritten = rewrite_json_record(
                record=rec,
                schema_hint="department_members",   # 或 "資工系師資名單"
                max_chars=400,
            )
        except Exception as e:
            print(f"rewrite_json_record 發生錯誤：{e}")
            sys.exit(1)

        # ===== 組成 Document =====
        content = rewritten.strip()

        metadata = {
            "source": source_path,
            "idx": i,
            "name": who["name"],
            "title": who["title"],
            "department": dept,
            "url": src_url,
            "content_type": "people",
        }

        docs.append(Document(page_content=content, metadata=metadata))

    return docs

# =========================
# news（系網新聞） adapter
# =========================
def _fmt_news_page_content(meta: Dict[str, Any], content: str) -> str:
    return "\n".join([
        f"類別：{meta.get('category','')}",   # ← 新增        
        f"標題：{meta.get('title','')}",
        f"日期：{meta.get('published_at','')}",
        f"連結：{meta.get('url','')}",        # ← 新增（方便 LLM/檢索知道來源）
        "內文：",
        content or "",
    ])

def news_records_to_documents(data: List[Dict[str, Any]], source_path: str) -> List[Document]:
    docs: List[Document] = []

    TARGET_CHARS = 1000
    OVERLAP_CHARS = 80

    total = len(data)
    print(f"[news] {source_path}：共 {total} 筆新聞，開始重寫…")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TARGET_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        separators=["\n\n", "\n", "。", "！", "？", "；", "、", "：", "——", " ", ",", ".", "，", ":"]
    )

    def to_ts(s: str | None) -> int | None:
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
            try:
                return int(datetime.strptime(s, fmt).timestamp())
            except Exception:
                pass
        return None

    for i, rec in enumerate(data, 1):
        if not isinstance(rec, dict):
            rec = {"value": rec}

        title = rec.get("title") or ""
        content = rec.get("content") or ""
        published_at = rec.get("published_at")
        published_ts = to_ts(published_at)
        url = rec.get("url")
        category = rec.get("category") or ""   # 對應 ttu_cse_news.sorted.json

        # 為每篇新聞產生穩定 article_id（利於重組與去重）
        article_key = f"{source_path}|{url or title}|{published_at or ''}|{i}"
        article_id = hashlib.sha1(article_key.encode("utf-8")).hexdigest()[:12]

        base_meta = {
            "source": source_path,
            "file_type": "json",

            "type": "news",
            "content_type": "news",

            "url": url,
            "title": title,
            "category": category,
            "published_at": published_at,
            "published_at_ts": published_ts,

            "idx": i,
            "article_id": article_id,
            "needs_split": False,
        }

        # === 情況一：內文長度不超過 TARGET_CHARS，整篇當一個 doc 重寫 ===
        if len(content) <= TARGET_CHARS:
            record: Dict[str, Any] = {
                "標題": title,
                "分類": category,
                "發布時間": published_at,
                "網址": url,
                "文章內容": content,
                "來源檔案": source_path,
                "article_id": article_id,
                "published_at_ts": published_ts,
            }

            try:
                rewritten = rewrite_json_record(
                    record=record,
                    schema_hint="news_article",
                    max_chars=TARGET_CHARS,
                )
            except Exception as e:
                print(f"[news_records_to_documents] rewrite_json_record 發生錯誤（程式終止）：{e}")
                sys.exit(1)

            docs.append(Document(
                page_content=rewritten.strip(),
                metadata=base_meta
            ))
            if i == 1 or i % 10 == 0 or i == total:
                print(f"[news] {source_path}：已完成 {i}/{total} 筆（短文）")
            continue

        # === 情況二：內文太長 → 先切成多個 chunk，再逐 chunk 重寫 ===
        parts = splitter.split_text(content)

        # 如果最後一塊太短，併回前一塊（你原本的邏輯）
        if len(parts) >= 2 and len(parts[-1]) < TARGET_CHARS // 3:
            parts[-2] = parts[-2] + ("\n" if not parts[-2].endswith("\n") else "") + parts[-1]
            parts.pop()

        for j, part in enumerate(parts):
            meta = dict(base_meta)
            meta.update({"chunk": j})

            # 針對「文章某一段」組成 record，讓 LLM 知道這是同一篇新聞的其中一部分
            record_chunk: Dict[str, Any] = {
                "標題": title,
                "分類": category,
                "發布時間": published_at,
                "網址": url,
                "文章內容片段": part,
                "所屬篇章 article_id": article_id,
                "chunk_index": j,
                "來源檔案": source_path,
                "published_at_ts": published_ts,
            }

            try:
                rewritten_chunk = rewrite_json_record(
                    record=record_chunk,
                    schema_hint="news_article_chunk",
                    max_chars=TARGET_CHARS,
                )
            except Exception as e:
                print(f"[news_records_to_documents] rewrite_json_record (chunk) 發生錯誤（程式終止）：{e}")
                sys.exit(1)

            docs.append(Document(
                page_content=rewritten_chunk.strip(),
                metadata=meta
            ))

            # 一篇長文所有 chunk 都處理完
            if i == 1 or i % 10 == 0 or i == total:
                print(f"[news] {source_path}：已完成 {i}/{total} 筆（長文多 chunk）")

    return docs



# =========================
# JSON 載入與分派
# =========================
def load_json_as_documents(path: Path) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    schema = detect_schema(obj)
    if schema == "people":
        # 1) 取出 member_list（支援多種 people JSON 格式）
        if isinstance(obj, dict) and "總覽" in obj:
            overview = obj["總覽"] or {}
            member_list = overview.get("成員列表", []) or []
        elif isinstance(obj, dict) and "成員列表" in obj:
            member_list = obj.get("成員列表", []) or []
        else:
            member_list = obj if isinstance(obj, list) else [obj]

        # 2) 個別老師 documents
        docs = people_records_to_documents(member_list, str(path))

        # 3) ✅ 教授總覽 documents（不管格式都加）
        docs.extend(
            people_overview_to_documents(member_list, str(path), max_chars=500)
        )

        return docs
    elif schema == "news":
        data = obj if isinstance(obj, list) else [obj]
        return news_records_to_documents(data, str(path))
    elif schema == "school":
        # 直接交給 school adapter 處理（不再 flatten）
        return school_info_to_documents(obj, str(path))
    elif schema == "academic_rules":
        data = obj if isinstance(obj, list) else [obj]
        return academic_records_to_documents(data, str(path))
    elif schema == "contacts":
        data = obj if isinstance(obj, list) else [obj]
        return contact_records_to_documents(data, str(path))
        # --- ✅ 新格式巢狀課程歷史 ---
    elif schema == "course_history_nested":
        return course_history_nested_to_documents(obj, str(path))
    elif schema == "course_overview":
        data = obj if isinstance(obj, list) else [obj]
        return course_overview_to_documents(data, str(path))
    elif schema == "program_courses":
        data = obj if isinstance(obj, list) else [obj]
        return program_courses_to_documents(data, str(path))
    elif schema == "calendar":
        data = obj if isinstance(obj, list) else [obj]
        docs = []
        docs.extend(calendar_months_to_documents(data, str(path)))   # 月總覽（原本的）
        docs.extend(calendar_events_to_documents(data, str(path)))   # ✅ 新增：單筆活動
        return docs


        # 其他已知 schema 都在上面處理完
    else:
        # --- 通用：用 LLM 先把「一筆 JSON 記錄」改寫成自然語句，再當成 doc ---
        # 正規化成 list
        if isinstance(obj, list):
            json_data = obj
        else:
            json_data = [obj]

        docs: List[Document] = []
        for idx, row in enumerate(json_data):
            if not isinstance(row, dict):
                row = {"value": row}

            text = rewrite_json_record(
                row,
                schema_hint=schema or path.stem,   # 例如 "faculty", "scholarship"
                max_chars=400,
            )

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),        # 這裡直接用 path 就好
                        "idx": idx,
                        "type": "json",
                        "schema": schema or "unknown",
                    },
                )
            )

        return docs


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
    ctype_count = Counter(
        d.metadata.get("content_type", d.metadata.get("type", "unknown"))
        for d in final_docs
    )
    print("▶ 類型統計：", dict(ctype_count))

    print("▶ 準備嵌入模型(多語)…")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        # model_kwargs={"device": "cpu"},
        model_kwargs={"device": "cuda"},
        # model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # 👈 跟 query.py 一樣
    )

    print("▶ 建立/更新 Chroma 向量庫…")
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"},   # 距離度量：cosine / l2 / ip
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
