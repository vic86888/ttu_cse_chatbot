# -*- coding: utf-8 -*-
import os
import re
import json
import urllib.parse
import ssl

import requests
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager
from bs4 import BeautifulSoup
import camelot

# 備註：這個不是必須，但可以用來從「表格外」再抓一次備註
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# =================================
#   自訂 TLS Adapter：關掉嚴格驗證
# =================================

class NonStrictTLSAdapter(HTTPAdapter):
    """
    使用較寬鬆的 X.509 驗證：
    - 保留一般 CA 驗證與主機名稱比對
    - 關閉 VERIFY_X509_STRICT（對某些舊憑證比較友善）
    """

    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        ctx = ssl.create_default_context()
        # 在較新版本 Python/OpenSSL 下才有 verify_flags / VERIFY_X509_STRICT
        if hasattr(ctx, "verify_flags") and hasattr(ssl, "VERIFY_X509_STRICT"):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kwargs["ssl_context"] = ctx
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **kwargs,
        )

# 建立共用的 Session，所有 HTTPS 都用這個 adapter
session = requests.Session()
session.mount("https://", NonStrictTLSAdapter())

# =========================
#  常數設定
# =========================

# 必修科目表頁面（不是直接 PDF）
PAGE_URL = "https://cse.ttu.edu.tw/p/412-1058-2032.php"
OUTPUT_PDF = "latest_cse_required.pdf"
OUTPUT_JSON = "cse_required_by_semester.json"

# 固定 title（照你給的）
TITLE_STATIC = "大同大學資訊工程學系大學部必修科目(檢核)表"

# 學期標籤
SEMESTER_LABELS = ["一上", "一下", "二上", "二下", "三上", "三下", "四上", "四下"]


# =========================
#  抓最新 PDF
# =========================

def find_latest_pdf_url(page_url: str) -> str | None:
    """
    從指定頁面找出「最新」的 PDF 連結並回傳絕對網址。
    規則：
    1. 找出所有 href 包含 '.pdf' 的 <a>
    2. 從連結文字抓 '114學年度' 這種格式的數字
    3. 優先選「學年度最大」的；都沒有就選第一個 PDF
    """
    print(f"⏳ 連線到：{page_url}")
    # 使用自訂 session，裡面已經套用 NonStrictTLSAdapter
    resp = session.get(page_url, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    pdf_links = []
    year_pattern = re.compile(r"(1[0-9]{2})學年度")  # e.g. 114學年度

    for idx, a in enumerate(soup.find_all("a", href=True)):
        href = a["href"]
        if ".pdf" not in href.lower():
            continue

        text = a.get_text(strip=True)
        m = year_pattern.search(text)
        year = int(m.group(1)) if m else None

        pdf_links.append(
            {
                "index": idx,
                "href": href,
                "text": text,
                "year": year,
            }
        )

    if not pdf_links:
        return None

    def sort_key(item):
        # 有寫學年度的比較優先
        has_year = item["year"] is not None
        year_val = item["year"] if item["year"] is not None else 0
        # index 越小（越前面）越優先
        return (has_year, year_val, -item["index"])

    best = sorted(pdf_links, key=sort_key, reverse=True)[0]
    pdf_url = urllib.parse.urljoin(page_url, best["href"])

    print(f"✔ 選到的連結文字：{best['text']}")
    print(f"✔ PDF URL：{pdf_url}")
    return pdf_url


def download_file(url: str, filename: str) -> str:
    """
    下載指定 URL 檔案到目前資料夾，回傳實際檔名
    """
    print(f"⏳ 下載 PDF：{url}")
    # 同樣使用共用 session
    resp = session.get(url, stream=True, timeout=20)
    resp.raise_for_status()

    with open(filename, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✔ 下載完成：{filename}")
    return filename


# =========================
#  學分與科目字串處理
# =========================

def parse_credit_from_raw(raw_text: str) -> int | None:
    """
    從 raw 裡面抓學分：
    例如 '國語文 能力表達一 (2)' -> 2 (int)
    抓不到就回傳 None
    """
    m = re.search(r"\((\d+)\)", raw_text)
    if m:
        return int(m.group(1))
    return None


def remove_trailing_credit(raw_text: str) -> str:
    """
    把最後面的 ' (2)'、' (3)' 這類括號 + 數字去掉
    '國語文 能力表達一 (2)' -> '國語文 能力表達一'
    """
    return re.sub(r"\s*\(\d+\)\s*$", "", raw_text).strip()


# =========================
#  從整份 PDF（表格外）再抓一次備註
# =========================

def extract_remarks_from_pdf(pdf_path: str) -> str:
    """
    從整份 PDF 文字中找出「備註」區塊（處理在表格外面的備註）。
    規則：
    - 找到第一行包含「備註」的文字（忽略空白）
    - 該行若有「備註：內容」，會把「內容」一起收集
    - 之後的非空行都視為備註，直到遇到第一個完全空白行為止
    """
    if pdfplumber is None:
        # 沒有安裝 pdfplumber 就直接放棄，不會擋住主流程
        print("⚠️ 尚未安裝 pdfplumber，無法從表格外擷取備註（可 pip install pdfplumber）")
        return ""

    remarks_lines: list[str] = []
    in_remarks = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                no_space = line.replace(" ", "")
                if not in_remarks and "備註" in no_space:
                    in_remarks = True
                    # 可能是「備註：1. xxxx」這種一行就有內容
                    after = no_space.split("備註", 1)[1]
                    after = after.lstrip("：:")
                    if after:
                        remarks_lines.append(after)
                    continue

                if in_remarks:
                    if line.strip():
                        remarks_lines.append(line.strip())
                    else:
                        # 遇到第一個完全空白行就視為備註結束
                        in_remarks = False
                        break

    return "\n".join(remarks_lines) if remarks_lines else ""


# =========================
#  Camelot 解析 + 轉成 JSON
# =========================

def pdf_to_semester_json(pdf_path: str, page_url: str, json_path: str) -> None:
    """
    用 Camelot 把 PDF 裡的表格抓出來，
    並依照「一上／一下／二上／二下／三上／三下／四上／四下」分組，
    產生 JSON schema：

    {
      "source_pdf": "https://cse.ttu.edu.tw/p/412-1058-2032.php",
      "title": "大同大學資訊工程學系大學部必修科目(檢核)表",
      "semesters": {
        "一上": [
          {
            "類別": "共同必修-語文能力",
            "raw": "國語文 能力表達一",
            "學分": 2,
            "共同必修小計": 15,
            "專業必修小計": 0
          },
          ...
        ],
        ...
      },
      "備註": "1. ...\n2. ...\n3. ..."
    }
    """

    print("⏳ 使用 Camelot 解析 PDF（lattice 模式）...")
    tables = camelot.read_pdf(
        pdf_path,
        pages="all",
        flavor="lattice",   # 若抓不到可以改 "stream" 試試
        strip_text="\n",
    )
    print(f"✔ 共抓到 {tables.n} 個表格")

    # 學期 → list of 科目
    courses: dict[str, list[dict]] = {label: [] for label in SEMESTER_LABELS}

    # 學分小計：key 是學期，value 是 int
    totals_common: dict[str, int | None] = {label: None for label in SEMESTER_LABELS}  # 基礎共同必修
    totals_major: dict[str, int | None] = {label: None for label in SEMESTER_LABELS}   # 系訂專業必修

    # ⭐ 備註：從「備註」那一行開始往下收集
    remarks_lines: list[str] = []
    in_remarks = False

    for t_idx, table in enumerate(tables):
        df = table.df.replace("\n", " ", regex=True)
        rows = df.values.tolist()

        # 1) 找出學期 header 在哪一列、每一個學期對應哪一欄
        semester_positions: dict[str, tuple[int, int]] = {}  # label -> (row_idx, col_idx)
        max_header_rows_to_scan = min(10, len(rows))

        for r in range(max_header_rows_to_scan):
            row = rows[r]
            for c, cell in enumerate(row):
                text = str(cell).strip()
                for label in SEMESTER_LABELS:
                    if text == label or text.startswith(label):
                        if label not in semester_positions:
                            semester_positions[label] = (r, c)

        if semester_positions:
            header_row = min(pos[0] for pos in semester_positions.values())
            semester_cols: dict[str, int] = {}
            for label, (r, c) in semester_positions.items():
                if r == header_row:
                    semester_cols[label] = c
        else:
            header_row = None
            semester_cols = {}

        print(f"表格 {t_idx} 在第 {table.page} 頁：學期欄位 = {semester_cols}")

        # 目前所在的大分類 / 小分類
        current_main_cat: str | None = None   # "共同必修" 或 "專業必修"
        current_sub_cat: str | None = None    # "語文能力"、"體育"、"公民素養" 之類

        for r, row in enumerate(rows):
            # 把這一列所有 cell 合併成一個字串，方便做關鍵字判斷
            row_text = " ".join(str(cell).strip() for cell in row if str(cell).strip())
            row_text_no_space = row_text.replace(" ", "")

            # ---- 備註偵測（抓 PDF 裡的備註）----
            if (not in_remarks) and "備註" in row_text_no_space:
                in_remarks = True
                # 這一列可能是「備註：1. xxxx」這種一行就有內容
                after = row_text_no_space.split("備註", 1)[1]
                after = after.lstrip("：:")
                if after:
                    remarks_lines.append(after)
                # 這一列只當備註標題＋內容，不當作科目列
                continue

            if in_remarks:
                # 備註內容通常是後面幾行條列式文字
                if row_text.strip():
                    remarks_lines.append(row_text.strip())
                    # 不要把備註當作科目列
                    continue
                else:
                    # 遇到完全空白的一行就結束備註區
                    in_remarks = False
                    continue

            # ---- 類別（共同必修 / 專業必修）偵測 ----
            if "基礎" in row_text and "共同" in row_text and "必修" in row_text:
                current_main_cat = "共同必修"
            elif "系訂" in row_text and "專業" in row_text and "必修" in row_text:
                current_main_cat = "專業必修"
                current_sub_cat = None  # 專業必修不再細分子類別

            # ---- 子類別偵測（語文能力 / 體育 / 公民素養）----
            if "語文" in row_text and "能力" in row_text:
                current_sub_cat = "語文能力"
            elif "體育" in row_text and "素養" not in row_text:
                current_sub_cat = "體育"
            elif "公民" in row_text and "素養" in row_text:
                current_sub_cat = "公民素養"

            # ---- 學分小計偵測 ----
            if semester_cols and "學分小計" in row_text and current_main_cat is not None:
                # 根據目前的大類別決定要填哪一組 totals
                target = totals_common if current_main_cat == "共同必修" else totals_major
                for label, col in semester_cols.items():
                    if col < len(row):
                        cell_text = str(row[col]).strip()
                        m = re.search(r"(\d+)", cell_text)
                        if m:
                            target[label] = int(m.group(1))
                # 這一列只當作小計，不當作科目列
                continue

            # ---- 如果沒有學期 header，就不用處理科目 ----
            if not semester_cols or header_row is None or r <= header_row:
                continue

            # ---- 真正的科目列：掃每一個學期欄位 ----
            for label, col in semester_cols.items():
                if col >= len(row):
                    continue
                cell_text = str(row[col]).strip()
                if not cell_text:
                    continue
                # 避免抓到 header 本身
                if cell_text in SEMESTER_LABELS:
                    continue
                # 避免只有 "(2)" 這種純學分
                if re.fullmatch(r"\(\d+\)", cell_text):
                    continue

                # 組出類別字串
                if current_main_cat == "共同必修":
                    if current_sub_cat:
                        category = "共同必修-" + current_sub_cat
                    else:
                        category = "共同必修"
                elif current_main_cat == "專業必修":
                    category = "專業必修"
                else:
                    category = "未分類"

                credit = parse_credit_from_raw(cell_text)
                raw_clean = remove_trailing_credit(cell_text)

                item = {
                    "類別": category,
                    "raw": raw_clean,   # ✅ 括號已拿掉
                    "學分": credit,     # ✅ 學分是從原始 cell_text 解析
                    # 小計稍後再補
                }
                courses[label].append(item)

    # ---- 把學分小計補回每一個科目物件 ----
    for label in SEMESTER_LABELS:
        common_total = totals_common.get(label)
        major_total = totals_major.get(label)
        for item in courses[label]:
            item["共同必修小計"] = common_total
            item["專業必修小計"] = major_total

    # ---- 組成最後的 JSON 結構 ----
    # 若在表格裡沒有抓到備註，就再用 pdfplumber 掃一次整份 PDF
    if not remarks_lines:
        extra_remarks = extract_remarks_from_pdf(pdf_path)
        if extra_remarks:
            remarks_text = extra_remarks
        else:
            remarks_text = ""
    else:
        remarks_text = "\n".join(remarks_lines)

    result = {
        "source_pdf": page_url,                 # 跟你範例一樣用頁面 URL
        "title": TITLE_STATIC,                  # 固定字串
        "semesters": courses,
        "備註": remarks_text,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✔ 依學期分組的 JSON 已輸出：{json_path}")


# =========================
#  main
# =========================

def main():
    # 1. 找並下載最新 PDF
    pdf_url = find_latest_pdf_url(PAGE_URL)
    if not pdf_url:
        print("✘ 找不到任何 PDF 連結 QQ")
        return

    download_file(pdf_url, OUTPUT_PDF)

    # 2. Camelot 解析 → 依「一上 / 一下 / ...」分組 → 產生 JSON
    pdf_to_semester_json(OUTPUT_PDF, PAGE_URL, OUTPUT_JSON)


if __name__ == "__main__":
    main()
