# -*- coding: utf-8 -*-
import json
import ssl
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager

# ==============================
#  SSL：放寬嚴格驗證（解決憑證問題）
# ==============================

class NonStrictTLSAdapter(HTTPAdapter):
    """
    使用較寬鬆的 X.509 驗證：
    - 保留一般 CA 驗證與主機名稱比對
    - 關閉 VERIFY_X509_STRICT（對部分舊憑證比較友善）
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

CONTINENT_COLOR = "rgb(102, 204, 204)"
COUNTRY_COLOR = "rgb(255, 255, 204)"


def scrape_ttu_partners():
    url = "https://ao.ttu.edu.tw/p/412-1073-2438.php"

    # 發送請求
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        # 嘗試用正確編碼解析（避免中文亂碼）
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"
    except requests.exceptions.SSLError as e:
        print(f"SSL 連線錯誤: {e}")
        return
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # 大同大學該網頁內容通常在 div.mcont 或 div.meditor 內
    content_div = (
        soup.find('div', class_='mcont')
        or soup.find('div', class_='meditor')
        or soup.body
    )

    if not content_div:
        print("無法找到主要內容區塊")
        return

    # 初始化資料結構：五大洲先建好（就算沒有資料也會出現在 JSON 裡）
    data = {
        "亞洲 (ASIA)": {},
        "歐洲 (EUROPE)": {},
        "美洲 (AMERICAS)": {},
        "大洋洲 (OCEANIA)": {},
        "非洲 (AFRICA)": {},
    }

    current_continent = None
    current_country = None
    started = False  # 進入姊妹校列表區域之後才開始吃 <a>
    last_href = None
    last_entry = None

    # 便利所有子孫節點（不限第一層，因為顏色/連結可能巢狀在裡面）
    for node in content_div.descendants:
        if not isinstance(node, Tag):
            continue

        style = node.get("style", "")

        # 1. 判斷「洲」：背景顏色 rgb(102, 204, 204)
        if CONTINENT_COLOR in style:
            text = node.get_text(" ", strip=True)
            if "亞洲" in text:
                current_continent = "亞洲 (ASIA)"
            elif "歐洲" in text:
                current_continent = "歐洲 (EUROPE)"
            elif "美洲" in text:
                current_continent = "美洲 (AMERICAS)"
            elif "大洋洲" in text:
                current_continent = "大洋洲 (OCEANIA)"
            elif "非洲" in text:
                current_continent = "非洲 (AFRICA)"
            else:
                current_continent = text or "未分類洲"

            if current_continent not in data:
                data[current_continent] = {}

            started = True
            current_country = None
            last_href = None
            last_entry = None
            continue

        # 2. 判斷「國家」：背景顏色 rgb(255, 255, 204)
        if COUNTRY_COLOR in style:
            text = node.get_text(" ", strip=True)
            current_country = text or "未分類國家"

            if current_continent is None:
                current_continent = "未分類洲"
                if current_continent not in data:
                    data[current_continent] = {}

            if current_country not in data[current_continent]:
                data[current_continent][current_country] = []

            last_href = None
            last_entry = None
            continue

        # 3. 處理學校超連結 <a>
        if node.name == "a":
            if not started or not current_continent:
                # 還沒進入正式列表（上面導航區的連結），略過
                continue

            school_name = node.get_text(strip=True)
            href = node.get("href", "").strip()
            if not school_name or not href:
                continue

            href_lower = href.lower()
            if href_lower.startswith("#") or href_lower.startswith("javascript"):
                continue

            full_url = urljoin(url, href)
            country_key = current_country or "其他地區 / Regional"

            if current_continent not in data:
                data[current_continent] = {}
            if country_key not in data[current_continent]:
                data[current_continent][country_key] = []

            # 同一個 href 連續兩個 <a>，當作名稱被拆開 → 合併
            if full_url == last_href and last_entry is not None:
                combined = f"{last_entry['name']} {school_name}"
                last_entry["name"] = " ".join(combined.split())
            else:
                school_info = {
                    "name": school_name,
                    "website": full_url,
                }
                data[current_continent][country_key].append(school_info)
                last_href = full_url
                last_entry = school_info

    # 把完全沒資料的洲拿掉（如果你想全部保留也可以註解掉這段）
    data = {k: v for k, v in data.items() if v}

    final_output = {
        "title": "大同大學姊妹校",
        "source": url,
        "continents": data,
    }

    print(json.dumps(final_output, indent=4, ensure_ascii=False))

    with open('ttu_sisters.json', 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        print("\n已儲存為 ttu_sisters.json")


if __name__ == "__main__":
    scrape_ttu_partners()
