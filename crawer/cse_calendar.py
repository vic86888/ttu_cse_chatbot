# 需求：pip install pymupdf camelot-py[cv] pandas requests beautifulsoup4 lxml
# Windows 若用 lattice 通常還需安裝 Ghostscript 與 OpenCV

import re
import json
from datetime import datetime
import fitz  # PyMuPDF
import camelot
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

# ========== 來源設定 ==========
# 只從頁面抓唯一 PDF
PAGE_URL = "https://reg.ttu.edu.tw/p/405-1032-23682,c910.php"
DOWNLOAD_DIR = Path(r"C:\Users\user\Python\CSE")

# ========== 裁切與擷取設定 ==========
PAGES = "all"          # Camelot 解析頁碼
LEFT_FRAC = 0.32       # 從左邊裁掉 32%
TOP_FRAC  = 0.10       # 從上方裁掉 10%

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# ---------------------------------------------------------
# 抓頁面上的唯一 PDF
# ---------------------------------------------------------
def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip()

def find_pdf_links(page_url: str) -> list[tuple[str, str]]:
    """回傳 [(pdf_url, link_text), ...]"""
    sess = requests.Session()
    r = sess.get(page_url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        abs_url = urljoin(page_url, href)
        if re.search(r"\.pdf(\?|$)", abs_url, flags=re.I):
            text = a.get_text(strip=True) or Path(unquote(urlparse(abs_url).path)).name
            links.append((abs_url, text))
    return links

def filename_from_response(resp: requests.Response, fallback: str) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, flags=re.I)
    if m:
        return safe_filename(unquote(m.group(1)))
    return safe_filename(fallback)

def download_pdf(pdf_url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url_name = Path(unquote(urlparse(pdf_url).path)).name or "download.pdf"
    with requests.get(pdf_url, headers={"User-Agent": UA}, stream=True, timeout=60) as r:
        r.raise_for_status()
        fname = filename_from_response(r, url_name)
        if not fname.lower().endswith(".pdf"):
            fname += ".pdf"
        out_path = out_dir / fname
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
    return out_path

def fetch_pdf_from_page(page_url: str, out_dir: Path):
    """
    回傳 (pdf_path, pdf_url)
    """
    links = find_pdf_links(page_url)
    if not links:
        raise SystemExit("此頁面未偵測到 PDF 連結，請確認網址。")

    # 網站只有一個 PDF，直接用第一個
    pdf_url, pdf_text = links[0]
    print(f"偵測到唯一 PDF：{pdf_text} -> {pdf_url}")

    path = download_pdf(pdf_url, out_dir)
    print(f"已下載：{path}")
    return path, pdf_url

# ---------------------------------------------------------
# 自動偵測學年度（每年更新不用改程式）
# ---------------------------------------------------------
def get_academic_years(pdf_path: Path):
    """
    回傳 (start_roc_year, end_roc_year)
    例：114學年度 -> (114, 115)
    偵測順序：
      1) 檔名
      2) PDF 第一頁文字
      3) 依今天日期推估
    """
    start_roc = None

    # 1) 從檔名抓 3 位數 ROC 年
    m = re.search(r'(?<!\d)(\d{3})\s*學年度', pdf_path.stem)
    if m:
        start_roc = int(m.group(1))

    # 2) 從 PDF 第一頁抓（用原始 PDF）
    if start_roc is None:
        try:
            doc = fitz.open(str(pdf_path))
            head_text = doc[0].get_text("text")
            doc.close()
            m = re.search(r'(?<!\d)(\d{3})\s*學年度', head_text)
            if m:
                start_roc = int(m.group(1))
        except:
            pass

    # 3) 最後 fallback：用今天推估
    if start_roc is None:
        today = datetime.today()
        roc_today = today.year - 1911
        # 學年度通常 8 月開始
        start_roc = roc_today if today.month >= 8 else roc_today - 1

    return start_roc, start_roc + 1

# ---------------------------------------------------------
# PDF 裁切
# ---------------------------------------------------------
def crop_left_top(src: Path, dst: Path, left_frac: float = 0.32, top_frac: float = 0.10):
    """
    從左邊裁掉 left_frac（例如 0.32 = 32%），同時從上方裁掉 top_frac（例如 0.10 = 10%）。
    """
    doc = fitz.open(str(src))
    out = fitz.open()

    for i in range(doc.page_count):
        r = doc[i].rect
        dx = min(max(r.width  * left_frac, 0), r.width  - 1)
        dy = min(max(r.height * top_frac, 0), r.height - 1)
        clip = fitz.Rect(r.x0 + dx, r.y0 + dy, r.x1, r.y1)

        np = out.new_page(width=clip.width, height=clip.height)
        np.show_pdf_page(np.rect, doc, i, clip=clip)

    out.save(str(dst))
    out.close()
    doc.close()

# ---------------------------------------------------------
# 讀表格（Camelot）
# ---------------------------------------------------------
def read_tables(path: Path):
    """優先用 lattice，抓不到再退回 stream。"""
    tables = camelot.read_pdf(
        str(path),
        flavor="lattice",
        pages=PAGES,
        strip_text="\n",
        line_scale=45
    )
    if getattr(tables, "n", 0) == 0:
        tables = camelot.read_pdf(
            str(path),
            flavor="stream",
            pages=PAGES,
            strip_text="\n",
            edge_tol=250,
            row_tol=10
        )
    return tables

# ===== 主流程 =====
def main():
    # 0) 從網頁下載唯一 PDF
    INPUT_PATH, PDF_URL = fetch_pdf_from_page(PAGE_URL, DOWNLOAD_DIR)
    assert Path(INPUT_PATH).exists(), f"找不到原始檔案：{INPUT_PATH}"

    # 自動偵測學年度
    start_year, end_year = get_academic_years(Path(INPUT_PATH))
    print(f"偵測學年度：{start_year}～{end_year}")

    # 裁切後 PDF 與輸出檔名（與來源同資料夾）
    CROPPED_PATH = Path(INPUT_PATH).with_name("cropped_left_32pct_top_10pct.pdf")
    OUT_JSON = Path(INPUT_PATH).with_name("calendar_last4.json")

    # 1) 先裁掉左 32% ＋ 上 10%
    crop_left_top(Path(INPUT_PATH), CROPPED_PATH, left_frac=LEFT_FRAC, top_frac=TOP_FRAC)
    print(f"已輸出裁切後 PDF：{CROPPED_PATH}")

    # 2) 讀表格
    tables = read_tables(CROPPED_PATH)
    print(f"偵測到表格數量：{getattr(tables, 'n', 0)}")

    # 3) 合併所有表格僅取最後四欄
    dfs = []
    for t in tables:
        df = t.df.copy()
        # 清空白列/欄
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        # 只取最後四欄
        if df.shape[1] >= 4:
            dfs.append(df.iloc[:, -4:])

    if not dfs:
        raise SystemExit("沒有偵測到可用的表格，可能需要調整參數或指定區域（table_areas）。")

    out = pd.concat(dfs, ignore_index=True)

    # 如果剛好四欄，就命名；否則沿用 Camelot 欄名
    if out.shape[1] == 4:
        out.columns = ["月", "日", "星期", "活動事項"]

    # ---- 4) 加上「年」欄位（動態學年度）----
    # 月份有些列會留白，先往下補齊
    out["月"] = out["月"].replace(r"^\s*$", pd.NA, regex=True).ffill()

    def infer_year_from_month(m):
        if m is None or pd.isna(m):
            return None
        s = str(m).strip()
        # 月可能是 "9" 或 "9~10" 或 "9/10" 這種，取第一段月份數字
        m0 = re.split(r"[~\-/\s]", s)[0]
        try:
            mi = int(m0)
        except ValueError:
            return None
        return start_year if mi >= 8 else end_year

    out["年"] = out["月"].apply(infer_year_from_month)

    # ---- 5) 轉成 JSON（欄位順序：title, 資料來源(頁面), 年, 月, 日, 星期, 活動事項）----
    out = out.where(pd.notna(out), None)
    for col in out.columns:
        out[col] = out[col].apply(lambda x: str(x).strip() if x is not None else None)

    raw_records = out.to_dict(orient="records")

    records = []
    for r in raw_records:
        records.append({
            "title": "行事曆", 
            "年": r.get("年"),
            "月": r.get("月"),
            "日": r.get("日"),
            "星期": r.get("星期"),
            "活動事項": r.get("活動事項"),
            "資料來源": PAGE_URL
        })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"完成：{OUT_JSON}")

if __name__ == "__main__":
    main()
