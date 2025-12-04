# -*- coding: utf-8 -*-
import json
import re
from io import BytesIO
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pdfplumber
import ssl
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager

PAGE_URL = "https://assistance.ttu.edu.tw/p/405-1035-12486,c642.php?"
BASE = "https://assistance.ttu.edu.tw"

# ==============================
#  SSL：寬鬆驗證（保留 CA/hostname，比預設鬆）
# ==============================

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

# 全域共用 session（所有 HTTPS 都套用上面的 adapter）
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; ttu-single-page-crawler/1.0)"
})
session.mount("https://", NonStrictTLSAdapter())


def extract_pdf_title_and_text(pdf_bytes: bytes):
    """
    從 PDF 取出：
    - title: 第一頁第一個非空行
    - full_text: 全部頁面的文字
    """
    title = ""
    text_all = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            txt = page.extract_text()
            if not txt:
                continue

            # 只在第一頁找標題
            if page_idx == 0 and not title:
                for line in txt.splitlines():
                    line = line.strip()
                    if line:
                        title = line
                        break

            text_all.append(txt)

    full_text = "\n".join(text_all)

    # 如果找不到標題，就讓 title 先是空字串，由外面決定 fallback
    return title, full_text


def split_by_items(text: str):
    """
    根據行首的 （一）（二）（三）… 分段
    回傳：
      prefix: (一) 以前的前言文字
      items:  [
        { "item": "一", "text": "整段內容" },
        ...
      ]
    """

    pattern = r'(?:^|\n)[（(]([一二三四五六七八九十]+)[）)]'
    parts = re.split(pattern, text, flags=re.MULTILINE)

    prefix = parts[0].strip()
    items = []

    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        label = parts[i]            # "一" / "二" / ...
        body = parts[i + 1].strip()

        # 保險再清一次開頭的 "(一)"
        body = re.sub(r'^[（(][一二三四五六七八九十]+[）)]\s*', '', body)

        items.append({
            "item": label,
            "text": body
        })

    return prefix, items


def main():
    # 1. 抓這一頁 HTML（用寬鬆驗證的 session）
    resp = session.get(PAGE_URL, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 頁面本身的大標題（可有可無）
    page_title = ""
    title_tag = soup.find(["h1", "h2", "h3"])
    if title_tag:
        page_title = title_tag.get_text(strip=True)

    # 2. 找 PDF 連結（只取第一個符合的）
    pdf_url = None
    pdf_link_text = None  # 超連結上顯示的文字（備查用）

    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").strip()
        href = a["href"]

        if ".pdf" in text.lower() or ".pdf" in href.lower() or "downloadfile" in href.lower():
            pdf_url = urljoin(BASE, href)
            pdf_link_text = text if text else "rules.pdf"
            break

    if not pdf_url:
        raise RuntimeError("在這一頁沒有找到 PDF 連結")

    print("找到 PDF：", pdf_link_text, "=>", pdf_url)

    # 3. 下載 PDF（同樣用 session）
    r = session.get(pdf_url, timeout=20)
    r.raise_for_status()

    # 4. 從 PDF 抓標題＋全文
    pdf_title, full_text = extract_pdf_title_and_text(r.content)

    # 如果 PDF 裡沒成功抓到標題，就退而求其次用超連結文字
    if not pdf_title:
        pdf_title = pdf_link_text

    # 5. 把標題從全文開頭拿掉，再來拆 (一)(二)(三)…
    text_for_split = full_text
    if pdf_title and text_for_split.startswith(pdf_title):
        text_for_split = text_for_split[len(pdf_title):].lstrip()

    prefix, items = split_by_items(text_for_split)

    # 6. 組成 JSON
    data = {
        # ✅ 這裡就是你要的：PDF 內文裡的標題
        "title": pdf_title,
        # 以下是補充資訊
        "page_title": page_title,          # HTML 頁面的標題
        "source_page": PAGE_URL,           # HTML 來源頁
        "pdf_url": pdf_url,                # PDF 下載 URL
        "pdf_link_text": pdf_link_text,    # 超連結上的原始文字（備查）
        "prefix": prefix,                  # 在(一)之前的文字（可能含一些說明）
        "items": items                     # 按 (一)(二)(三)… 切好的段落
    }

    # 7. 存檔
    out_file = "ttu_single_page_rules.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已輸出 JSON 到 {out_file}")
    print(f"PDF 標題：{pdf_title}")
    print(f"共 {len(items)} 段 (一)(二)(三)…")


if __name__ == "__main__":
    main()
