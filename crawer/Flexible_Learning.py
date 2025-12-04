# -*- coding: utf-8 -*-
import io
import json
import re
import ssl
from pathlib import Path
from urllib.parse import urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager

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

BASE_URL = "https://tturule.ttu.edu.tw/rulelist/showcontent.php?rid=1008"
OUTPUT_PATH = Path("ttu_flexible_week.json")  # 輸出的 JSON 檔名


def get_page_soup(url: str) -> BeautifulSoup:
    """下載網頁並回傳 BeautifulSoup 物件"""
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    # 自動偵測編碼，避免中文亂碼
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    return BeautifulSoup(resp.text, "html.parser")


def extract_impl_points(soup: BeautifulSoup) -> str:
    """
    從整頁文字中，抓出「實施要點」條文區塊。
    """
    full_text = soup.get_text("\n", strip=True)

    title_kw = "大同大學數位教學課程實施要點"
    attach_kw = "附件檔案"

    # 找到標題所在位置
    title_pos = full_text.find(title_kw)
    if title_pos == -1:
        # 找不到就直接回傳全文讓你自己再手動處理
        return full_text

    # 從標題之後開始找「1.」當作第一條的起點（找不到就從標題後開始）
    num_pos = full_text.find("1.", title_pos)
    start_pos = num_pos if num_pos != -1 else title_pos

    # 以「附件檔案」當作尾巴（找不到就到全文結尾）
    end_pos = full_text.find(attach_kw, start_pos)
    if end_pos == -1:
        impl_text = full_text[start_pos:]
    else:
        impl_text = full_text[start_pos:end_pos]

    return impl_text.strip()


def find_pdf_url(soup: BeautifulSoup, base_url: str) -> str | None:
    """從頁面上找出規章 PDF 的連結（ruledl.php...）"""
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "ruledl.php" in href:
            return urljoin(base_url, href)
    return None


def extract_flexible_week_rules(pdf_url: str) -> str:
    """下載 PDF 並抽出「彈性教學週活動規劃」那一段原始文字。"""
    resp = session.get(pdf_url, timeout=20)
    resp.raise_for_status()
    pdf_bytes = resp.content

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        all_text = "\n".join((page.extract_text() or "") for page in pdf.pages)

    start_kw = "彈性教學週活動規劃"
    end_kw = "四、教學方式"  # 該段後面緊接的下一個大標題

    start = all_text.find(start_kw)
    if start == -1:
        raise ValueError("找不到『彈性教學週活動規劃』這段文字，可能 PDF 版面有變動。")

    end = all_text.find(end_kw, start)
    if end == -1:
        section = all_text[start:]
    else:
        section = all_text[start:end]

    return section.strip()


def extract_flexible_week_bullets(flexible_text: str):
    """
    從『彈性教學週活動規劃』區塊裡，只抓列點回傳 list。
    """
    lines = [line.strip() for line in flexible_text.splitlines() if line.strip()]
    bullets = []

    for line in lines:
        stripped = line.lstrip()
        # 依實際 PDF 的列點符號來判斷（、•、● 或 - 等）
        if stripped.startswith("") or stripped.startswith("•") or stripped.startswith("●") or stripped.startswith("-"):
            content = stripped[1:].strip()  # 去掉列點符號
            bullets.append(content)

    # 如果因為 PDF 變動導致沒抓到，可用關鍵字當 fallback
    if not bullets:
        for line in lines:
            if any(kw in line for kw in ("辦理時間", "上課時數", "彈性教學週 2 週時數")):
                bullets.append(line.strip())

    return bullets


def split_impl_points(impl_text: str):
    """
    把「實施要點」用 1. 2. 3. ... 一點一點切開。
    回傳格式範例：
    ["1. ...", "2. ...", "3. ..."]
    """
    text = impl_text.strip()
    # 用「數字 + .」當作分隔標記（但保留在結果裡）
    parts = re.split(r'(\n?\s*\d+\.)', text)

    items = []
    current = ""

    for part in parts:
        if not part.strip():
            continue

        if re.match(r'^\s*\d+\.$', part.strip()):
            # 遇到新的「N.」就開啟新的一點
            if current:
                items.append(current.strip())
            current = part.strip()  # 先把 "1." 放進去
        else:
            # 把內容接到目前這一點後面
            if current:
                # 保留原內文的換行
                current += " " + part.strip()
            else:
                # 若在第一個編號前的文字就略過（通常是標題）
                continue

    if current:
        items.append(current.strip())

    return items


def main():
    # 1. 下載主頁並解析
    soup = get_page_soup(BASE_URL)

    # 抓實施要點條文（整段）
    impl_text = extract_impl_points(soup)

    # 把實施要點切成「一點一筆」
    impl_items = split_impl_points(impl_text)

    # 2. 找出 PDF 下載網址
    pdf_url = find_pdf_url(soup, BASE_URL)
    if not pdf_url:
        print("找不到 PDF 連結，請檢查網頁是否有變動。")
        return

    # 3. 抽出 PDF 裡的「彈性教學週活動規劃」全文
    flexible_rules = extract_flexible_week_rules(pdf_url)

    # 4. 從中只取「列點」部分
    flexible_bullets = extract_flexible_week_bullets(flexible_rules)

    # 5. 組成 JSON 資料結構
    data = {
        "title": "大同大學數位教學課程實施要點及彈性教學週相關規定",
        "source_url": BASE_URL,
        "pdf_url": pdf_url,
        "實施要點": impl_items,
        "彈性教學週活動規劃": flexible_bullets,
    }

    # 6. 寫入 JSON 檔
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已輸出 JSON 檔：{OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
