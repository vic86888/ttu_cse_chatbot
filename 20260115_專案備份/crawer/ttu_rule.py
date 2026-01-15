# -*- coding: utf-8 -*-
import json
import re
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
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


URLS = [
    "https://tturule.ttu.edu.tw/rulelist/showcontent.php?rid=33",
    "https://cse.ttu.edu.tw/p/404-1058-2974.php",
    "https://activity.ttu.edu.tw/p/405-1036-4940.php?",
    "https://tturule.ttu.edu.tw/rulelist/showcontent.php?rid=329",
    "https://cse.ttu.edu.tw/p/404-1058-35967.php",
]

# 標題可能出現的關鍵字：辦法 / 要點 / 規定 / 章程 ...
TITLE_KEYWORDS = ("辦法", "要點", "規定", "章程")

# 條文抬頭可能的格式：第X條、一、二、三、、（一）、(一) ...
ARTICLE_PATTERNS = [
    # 支援「第 二 條」中間有空白
    re.compile(r"^第\s*[一二三四五六七八九十零百千0-9]+\s*條"),
    re.compile(r"^[一二三四五六七八九十]+、"),
    re.compile(r"^（[一二三四五六七八九十]+）"),
    re.compile(r"^\([一二三四五六七八九十]+\)"),
]


def make_filename(url: str) -> str:
    p = urlparse(url)
    qs = parse_qs(p.query)

    if "rid" in qs and qs["rid"]:
        rid = qs["rid"][0]
        return f"rule_{rid}.json"

    path_last = p.path.rstrip("/").split("/")[-1] or "index"
    safe_path = path_last.replace(".", "_")
    return f"{p.netloc}_{safe_path}.json"


def match_article_header(line: str):
    """檢查一行是不是條文抬頭（第X條、一、、（一）、(一)...）"""
    for pat in ARTICLE_PATTERNS:
        m = pat.match(line)
        if m:
            return m
    return None


def detect_title(lines):
    """
    從內容區的純文字行裡判斷標題：
    1. 找出所有包含 辦法/要點/規定/章程 的行
    2. 優先挑「同時包含『大同大學』」的那一行
    3. 再不行就拿第一個符合關鍵字的行
    4. 都沒有就拿第一行
    回傳：(title_idx, title_text)
    """
    candidate_indices = [
        i for i, line in enumerate(lines)
        if any(k in line for k in TITLE_KEYWORDS) and "附件" not in line
    ]

    # 優先：含「大同大學」的那行
    for i in candidate_indices:
        if "大同大學" in lines[i]:
            title_text = lines[i].lstrip("#").strip()
            return i, title_text

    # 次優先：第一個符合關鍵字的行
    if candidate_indices:
        idx = candidate_indices[0]
        title_text = lines[idx].lstrip("#").strip()
        return idx, title_text

    # 最後：fallback 到第一行
    if lines:
        return 0, lines[0]
    else:
        return -1, ""


def extract_rule_from_html(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # ✅ 只取「內容區」：.module-detail 裡面才是文章本體
    content_root = soup.select_one(".module-detail") or soup

    # 從內容區取出純文字
    text = content_root.get_text("\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 1. 判斷標題
    title_idx, title = detect_title(lines)

    # 2. 找條文開始：從標題之後，第一個 match 到條文抬頭的行
    start_idx = None
    for i in range(title_idx + 1, len(lines)):
        if match_article_header(lines[i]):
            start_idx = i
            break

    if start_idx is None:
        start_idx = title_idx + 1

    # 3. 找條文結束：遇到「附件」就停止
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if lines[i].startswith("附件"):
            end_idx = i
            break

    article_lines = lines[start_idx:end_idx]

    # 4. 切條文
    articles = []
    current_article = None

    for line in article_lines:
        m = match_article_header(line)
        if m:
            # 開始新條文
            if current_article:
                articles.append(current_article)

            article_no_raw = m.group(0)
            # 把「第 二 條」這種空白去掉 -> 「第二條」
            article_no = re.sub(r"\s+", "", article_no_raw)

            rest = line[m.end():].strip()

            current_article = {
                "article_no": article_no,
                "text": rest,
            }
        else:
            # 接在目前條文後面
            if current_article:
                if current_article["text"]:
                    current_article["text"] += "\n" + line
                else:
                    current_article["text"] = line
            else:
                # 前言先略過（需要可以額外存成 preface）
                pass

    if current_article:
        articles.append(current_article)

    return {
        "url": url,
        "title": title,
        "articles": articles,
    }


def crawl_and_save():
    for url in URLS:
        print(f"抓取：{url}")
        try:
            # 用寬鬆驗證的 session
            resp = session.get(url, timeout=10)
        except requests.exceptions.SSLError as e:
            print(f"SSL 連線錯誤：{e}")
            continue

        if resp.status_code != 200:
            print(f"！！抓取失敗 {resp.status_code}：{url}")
            continue

        if resp.apparent_encoding:
            resp.encoding = resp.apparent_encoding
        else:
            resp.encoding = "utf-8"

        data = extract_rule_from_html(resp.text, url)
        filename = make_filename(url)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已輸出：{filename}")


if __name__ == "__main__":
    crawl_and_save()
