# -*- coding: utf-8 -*-
import json
import re
from io import BytesIO

import requests
import docx
import ssl
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager

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
session.mount("https://", NonStrictTLSAdapter())

# 固定來源頁面與 DOCX 位置
SOURCE_PAGE = "https://cse.ttu.edu.tw/p/412-1058-3275.php?"
DOCX_URL = "https://cse.ttu.edu.tw/var/file/58/1058/img/104/545079179.docx"

OUTPUT_JSON_PATH = "cse_shishi_banfa.json"  # JSON 輸出檔名

# 條文 regex，用來切「第X條」
ARTICLE_PATTERN = re.compile(
    r"^第\s*[一二三四五六七八九十零百千0-9]+\s*條",
    re.MULTILINE
)


def split_articles(text: str):
    """
    用「第 X 條」把全文切成一條一條的條文。

    回傳格式:
    [
      {
        "heading": "第一條",
        "body": "這條的內容..."
      },
      ...
    ]
    """
    articles = []
    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        return articles  # 找不到就回傳空陣列

    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        block = text[start:end].strip()
        raw_heading = m.group().strip()
        # 把「第 一 條」這種中間多的空白拿掉，變成「第一條」
        heading = re.sub(r"\s+", "", raw_heading)

        # heading 之後的內容就是 body
        body = block[len(m.group()):].strip()

        articles.append({
            "heading": heading,
            "body": body
        })

    return articles


def extract_from_docx_bytes(content: bytes):
    """
    從 DOCX 的二進位內容中：
    1. 抓出完整標題：「大同大學資訊工程系(所)學生修讀學、碩士 五年一貫學程辦法」
       （中間就算有換行或多空白也會合併成一行）
    2. 用「第X條」切成多條條文。
    """
    document = docx.Document(BytesIO(content))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

    if not paragraphs:
        return {
            "title": "",
            "articles": []
        }

    # 把所有段落串成一個大字串，方便跨行搜尋
    full_text = "\n".join(paragraphs)

    # 針對完整標題做 regex，比較穩
    # \s* 可以吃掉中間的換行或多空白
    title_pattern = re.compile(
        r"大同大學資訊工程系\(所\)學生修讀學、碩士\s*五年一貫學程辦法"
    )
    m = title_pattern.search(full_text)

    if m:
        title_raw = m.group(0)
    else:
        # fallback 1：找第一個包含「五年一貫學程辦法」的段落
        title_raw = ""
        for txt in paragraphs:
            if "五年一貫學程辦法" in txt or "五年一貫學程辦法" in txt.replace(" ", ""):
                title_raw = txt
                break

        # fallback 2：找第一個包含「辦法」的段落
        if not title_raw:
            for txt in paragraphs:
                if "辦法" in txt:
                    title_raw = txt
                    break

        # fallback 3：最後就用第一段
        if not title_raw:
            title_raw = paragraphs[0]

    # 把換行、多重空白通通壓成一個空白
    title = re.sub(r"\s+", " ", title_raw).strip()

    # 條文直接對整份文字做 split（regex 會自己找到「第X條」開頭）
    articles = split_articles(full_text)

    return {
        "title": title,
        "articles": articles
    }


def main():
    # 下載 DOCX（用寬鬆驗證 session）
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; docx-scraper/1.0)"
    }
    resp = session.get(DOCX_URL, headers=headers, timeout=20)
    resp.raise_for_status()

    parsed = extract_from_docx_bytes(resp.content)

    # file_name 從 URL 取最後一段
    file_name = DOCX_URL.split("/")[-1]

    # 照你指定的格式輸出
    output = {
        "source_page": SOURCE_PAGE,
        "title": parsed["title"],
        "file_name": file_name,
        "file_url": DOCX_URL,
        "articles": parsed["articles"]
    }

    # 寫出 JSON 檔
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 也順便印在 console
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nJSON 已輸出到: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
