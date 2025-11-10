# 需求：pip install pymupdf camelot-py[cv] pandas requests beautifulsoup4 lxml
# Windows 若用 lattice 通常還需安裝 Ghostscript 與 OpenCV

import re
import fitz  # PyMuPDF
import camelot
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

# ========== 來源設定 ==========
PAGE_URL = "https://reg.ttu.edu.tw/p/405-1032-23682,c910.php"
DOWNLOAD_DIR = Path(r"C:\Users\user\Python\CSE")
LOCAL_INPUT_PATH = Path(r"C:\Users\user\Python\CSE\pta_38205_5343400_24465.pdf")

# ========== 裁切與擷取設定 ==========
PAGES = "all"
LEFT_FRAC = 0.32
TOP_FRAC  = 0.10

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# ---------------- 安全降級封裝（重點） ----------------
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from requests.exceptions import SSLError, RequestException
import urllib3, sys

# 只在這些有問題的網域允許 verify=False 降級
BAD_SSL_HOSTS = {"reg.ttu.edu.tw", "cse.ttu.edu.tw"}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retries = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

SESSION = build_session()

def safe_get(url: str, **kwargs) -> requests.Response:
    """
    先用 verify=True；若遇 SSLError 且主機在 BAD_SSL_HOSTS，才以 verify=False 降級重試一次。
    會從 kwargs 取出呼叫端的 timeout/verify，避免重複傳參數。
    """
    host = urlparse(url).hostname or ""
    timeout = kwargs.pop("timeout", 20)       # 取用戶指定的 timeout，否則預設 20
    verify = kwargs.pop("verify", True)       # 預設驗證憑證

    try:
        return SESSION.get(url, timeout=timeout, verify=verify, **kwargs)
    except SSLError as e:
        if host in BAD_SSL_HOSTS:
            sys.stderr.write(f"[warn] SSL 驗證失敗({host}) → verify=False 降級重試：{e}\n")
            return SESSION.get(url, timeout=timeout, verify=False, **kwargs)
        raise

# ---------------------------------------------------------
# 抓頁面上的 PDF
# ---------------------------------------------------------
def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip()

def find_pdf_links(page_url: str) -> list[tuple[str, str]]:
    """回傳 [(pdf_url, link_text), ...]"""
    r = safe_get(page_url, headers={"User-Agent": UA})
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
    with safe_get(pdf_url, headers={"User-Agent": UA}, stream=True, timeout=60) as r:
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

def fetch_pdf_from_page(page_url: str, out_dir: Path) -> Path:
    links = find_pdf_links(page_url)
    if not links:
        raise SystemExit("此頁面未偵測到 PDF 連結，請確認網址或改用本地檔。")
    first_url, first_text = links[0]
    print(f"偵測到 PDF：{first_text} -> {first_url}")
    path = download_pdf(first_url, out_dir)
    print(f"已下載：{path}")
    return path

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
    tables = camelot.read_pdf(str(path), flavor="lattice", pages=PAGES,
                              strip_text="\n", line_scale=45)
    if getattr(tables, "n", 0) == 0:
        tables = camelot.read_pdf(str(path), flavor="stream", pages=PAGES,
                                  strip_text="\n", edge_tol=250, row_tol=10)
    return tables

# ===== 主流程 =====
if PAGE_URL:
    INPUT_PATH = fetch_pdf_from_page(PAGE_URL, DOWNLOAD_DIR)
else:
    INPUT_PATH = LOCAL_INPUT_PATH

assert Path(INPUT_PATH).exists(), f"找不到原始檔案：{INPUT_PATH}"

CROPPED_PATH = Path(INPUT_PATH).with_name("cropped_left_32pct_top_10pct.pdf")
OUT_CSV = Path(INPUT_PATH).with_name("calendar_last4.csv")

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
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if df.shape[1] >= 4:
        dfs.append(df.iloc[:, -4:])

if not dfs:
    raise SystemExit("沒有偵測到可用的表格，可能需要調整參數或指定區域（table_areas）。")

out = pd.concat(dfs, ignore_index=True)

if out.shape[1] == 4:
    out.columns = ["月", "日", "星期", "活動事項"]

# 4) 輸出
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"完成：{OUT_CSV}")
