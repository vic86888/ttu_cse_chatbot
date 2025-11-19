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
    回傳 "people" / "news" / "school" / "unknown"
    - people: 有「人物」「電話」「信箱」等鍵
    - news:   有 "url","title","published_at","content"
    - school: 有「名稱」「英文名稱」「校訓」等鍵（例如 about_schoo.json）
    """
    sample = None
    if isinstance(obj, list) and obj:
        sample = obj[0]
    elif isinstance(obj, dict):
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
    return "unknown"

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
        year_term_raw = str(rec.get("學年學期", "")).strip()
        year, term = parse_year_term(year_term_raw)

        select_type = str(rec.get("選別", "")).strip()   # 必修 / 選修
        grade = str(rec.get("所屬年級", "")).strip()     # 一年級 / 二年級 / 三年級 / 四年級
        data_source = str(rec.get("資料來源", "")).strip()  # URL

        names = rec.get("課程名稱") or []
        if not isinstance(names, list):
            names = [names]
        # 清掉空字串，全部轉成 str
        names = [str(n).strip() for n in names if str(n).strip()]
        course_count = len(names)
        courses_str = "、".join(names)

        # 給 LLM 的文字內容
        lines = [
            f"學年學期:{year_term_raw}",
            f"所屬年級：{grade}",
            f"選別：{select_type}",
            "課程名稱列表：",
        ]
        lines.extend([f"- {n}" for n in names])
        if data_source:
            lines.append(f"資料來源：{data_source}")
        text = "\n".join(lines)

        meta = {
            "source": source_path,
            "file_type": "json",
            "type": "course_overview",        # 給統計/除錯用
            "content_type": "course_overview",# 之後 filter 用這個

            "year_term": year_term_raw,
            "year": year,                     # int 或 None
            "term": term,                     # "上" / "下" / ""
            "grade": grade,                   # 一年級 / 二年級 / 三年級 / 四年級

            "select_type": select_type,       # 必修 / 選修
            "course_count": course_count,     # int
            "courses": courses_str,           # ✅ 字串，不是 list
            "data_source": data_source,       # 資料來源 URL

            "idx": i,
            "needs_split": False,             # 不再切塊
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
        
        if "辦理項目" in rec:  # 行政/招生類
            role = "service"
            item = rec.get("辦理項目", "").strip()
            person = rec.get("承辦人", "").strip()
            ext = rec.get("分機", "").strip()

            lines = [
                f"辦理項目：{item}",
                f"承辦人：{person}",
                f"聯絡電話：{ext}",
            ]
        elif "學系" in rec:   # 各學系聯絡人
            role = "department"
            dept = rec.get("學系", "").strip()
            person = rec.get("聯絡人員", "").strip()
            ext = rec.get("分機", "").strip()

            lines = [
                f"學系：{dept}",
                f"聯絡人員：{person}",
                f"聯絡電話：{ext}",
            ]
        else:
            # 保險：不符合預期欄位就 flatten 一下
            role = "unknown"
            lines = [f"{k}：{v}" for k, v in rec.items() if k != "資料來源"]

        if data_source:
            lines.append(f"資料來源：{data_source}")
        text = "\n".join(lines)

        meta = {
            "source": source_path,
            "file_type": "json",
            "type": "contact",           # 給你統計用
            "content_type": "contact",   # 之後 filter 用這個
            "role": role,                # "service" or "department"
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
        category = rec.get("類別", "").strip()
        topic = infer_topic(category)

        lines = [f"類別：{category}"]
        for k, v in rec.items():
            if k == "類別":
                continue
            # 統一成「欄名：內容」的格式
            lines.append(f"{k}：{v}")
        text = "\n".join(lines)

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
    將 about_schoo.json 這類「學校資訊」整理成一份 Document，
    並在 metadata 裡補充常用欄位（校名、校訓、網址等）。
    """
    # 預期格式：list[dict]
    if not isinstance(obj, list) or not obj:
        text = str(obj)
        meta = {
            "source": source_path,
            "file_type": "json",
            "content_type": "school",
            "needs_split": False,
        }
        return [Document(page_content=text, metadata=meta)]

    # 做個安全的 helper：從不同 block 抓 key
    def find_key(key: str, default: str = "") -> Any:
        for block in obj:
            if isinstance(block, dict) and key in block:
                return block[key]
        return default

    # 基本資訊
    name = find_key("名稱", "")
    name_en = find_key("英文名稱", "")
    motto = find_key("校訓", "")
    founded_at = find_key("成立時間", "")
    founder = find_key("創辦人", "")
    school_type = find_key("類型", "")

    address = find_key("地址", "")
    phone = find_key("電話", "")
    emergency_phone = find_key("緊急校安專線", "")
    fax = find_key("傳真", "")
    president_phone = find_key("校長室電話", "")
    president_fax = find_key("校長室傳真", "")
    president_email = find_key("校長室 email", "")

    school_code = find_key("學校代碼", "")
    url = find_key("網址", "")
    departments = find_key("系所結構", [])
    if isinstance(departments, list):
        departments_str = "、".join(map(str, departments))
    else:
        departments_str = str(departments) if departments else ""

    prev_name = find_key("前身", "")
    reorg_at = find_key("改制時間", "")
    rename_at = find_key("更名時間", "")
    feature = find_key("特色", "")

    # 給 LLM 看的文字內容（你可以之後再微調格式）
    lines = [
        f"名稱：{name}",
        f"英文名稱：{name_en}",
        f"校訓：{motto}",
        f"成立時間：{founded_at}",
        f"創辦人：{founder}",
        f"類型：{school_type}",
        "",
        f"地址：{address}",
        f"電話：{phone}",
        f"緊急校安專線：{emergency_phone}",
        f"傳真：{fax}",
        f"校長室電話：{president_phone}",
        f"校長室傳真：{president_fax}",
        f"校長室 email：{president_email}",
        "",
        f"學校代碼：{school_code}",
        f"網址：{url}",
        f"系所結構：{departments_str}",
        "",
        f"前身：{prev_name}",
        f"改制時間：{reorg_at}",
        f"更名時間：{rename_at}",
        f"特色：{feature}",
    ]
    text = "\n".join(lines)

    meta = {
        "source": source_path,
        "file_type": "json",
        "content_type": "school",
        "name": name,
        "name_en": name_en,
        "motto": motto,
        "founded_at": founded_at,
        "founder": founder,
        "school_type": school_type,
        "address": address,
        "phone": phone,
        "emergency_phone": emergency_phone,
        "fax": fax,
        "president_phone": president_phone,
        "president_fax": president_fax,
        "president_email": president_email,
        "school_code": school_code,
        "url": url,
        "departments": departments_str,   # ✅ 存成純字串就沒問題
        "prev_name": prev_name,
        "reorg_at": reorg_at,
        "rename_at": rename_at,
        "feature": feature,
        "needs_split": False,  # 這份本來就不長，不再二次切塊
        "idx": 1,
    }

    return [Document(page_content=text, metadata=meta)]

# =========================
# people（老師名錄） adapter
# =========================
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


def _split_meta(raw: str) -> Dict[str, str]:
    """
    把原本塞在 metadata 的字串拆成三塊：
    - education: 學歷
    - experience: 經歷
    - expertise: 教學與研究領域
    這樣就不會在「研究領域」裡再把學歷、經歷重複印一次。
    """
    raw = (raw or "").strip()
    if not raw:
        return {"education": "", "experience": "", "expertise": ""}

    education = ""
    experience = ""
    expertise = ""

    txt = raw

    # 先切掉「教學與研究領域」那一段，剩下前面給學歷/經歷用
    head, sep, tail = txt.partition("教學與研究領域")
    if sep:  # 有找到教學與研究領域
        txt = head.strip()
        expertise = tail.lstrip(" ：:").strip()
    else:
        txt = raw

    # 處理學歷 / 經歷
    if "學歷" in txt or "經歷" in txt:
        if "學歷" in txt:
            after_degree = txt.split("學歷", 1)[1]
            after_degree = after_degree.lstrip(" ：:").strip()
        else:
            after_degree = txt

        if "經歷" in after_degree:
            part_deg, part_exp = after_degree.split("經歷", 1)
            education = part_deg.strip(" 。\n\r\t")
            experience = part_exp.lstrip(" ：:").strip()
        else:
            education = after_degree.strip(" 。\n\r\t")
    else:
        # 沒有特別標學歷/經歷，就全部當成研究/教學說明
        if not expertise:
            expertise = raw

    return {
        "education": education,
        "experience": experience,
        "expertise": expertise,
    }


def _fmt_people_page_content(meta: Dict[str, Any]) -> str:
    lines = [
        f"姓名：{meta.get('name','')}",
        f"職稱/職務：{meta.get('title','')}",
    ]
    if meta.get("department"):
        lines.append(f"系所：{meta['department']}")
    lines.extend([
        f"辦公室：{meta.get('office','')}",
        f"分機/電話：{meta.get('phone','')}",
        f"Email：{meta.get('email','')}",
    ])
    if meta.get("education"):
        lines.append(f"學歷：{meta['education']}")
    if meta.get("experience"):
        lines.append(f"經歷：{meta['experience']}")
    if meta.get("expertise"):
        lines.append(f"研究領域：{meta['expertise']}")
    if meta.get("data_source"):
        lines.append(f"資料來源：{meta['data_source']}")
    return "\n".join(lines)


def people_records_to_documents(
    data: List[Dict[str, Any]], source_path: str
) -> List[Document]:
    docs: List[Document] = []
    for i, rec in enumerate(data, 1):
        # 支援兩種格式:
        # 1. 舊格式: 「人物」欄位包含姓名和職稱
        # 2. 新格式: 「姓名」和「職稱」分開
        if "姓名" in rec:
            # 新格式 (department_members.json)
            who = {"name": rec.get("姓名", "").strip(), "title": rec.get("職稱", "").strip()}
        else:
            # 舊格式
            who = _parse_name_title(rec.get("人物", ""))

        # 取得系所和資料來源
        department = rec.get("系所", "") or ""
        data_source = rec.get("資料來源", "") or ""

        raw_meta = rec.get("metadata") or ""
        meta_parsed = _split_meta(raw_meta)

        meta = {
            "source": source_path,
            "file_type": "json",
            "content_type": "people",
            "name": who["name"],
            "title": who["title"],
            "phone": rec.get("電話"),
            "email": rec.get("信箱"),
            "office": rec.get("辦公室"),
            "department": department,
            "data_source": data_source,
            "education": meta_parsed["education"],
            "experience": meta_parsed["experience"],
            "expertise": meta_parsed["expertise"],
            "idx": i,
            "needs_split": False,
        }
        docs.append(
            Document(
                page_content=_fmt_people_page_content(meta),
                metadata=meta,
            )
        )
    return docs

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
    elif schema == "school":
        # 直接交給 school adapter 處理（不再 flatten）
        return school_info_to_documents(obj, str(path))
    elif schema == "academic_rules":
        data = obj if isinstance(obj, list) else [obj]
        return academic_records_to_documents(data, str(path))
    elif schema == "contacts":
        data = obj if isinstance(obj, list) else [obj]
        return contact_records_to_documents(data, str(path))
    elif schema == "course_history":
        data = obj if isinstance(obj, list) else [obj]
        return course_records_to_documents(data, str(path))
    elif schema == "course_overview":
        data = obj if isinstance(obj, list) else [obj]
        return course_overview_to_documents(data, str(path))
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
