# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import re
import json
import urllib3

# 關掉因 verify=False 產生的警告（只建議在你確定信任的站台時這樣做）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def parse_section2_table(lines):
    """
    解析「二、請依照下列步驟進行申請程序」底下的表格：
    欄位固定為：編號 / 項目 / 說明
    回傳 list[ { 編號, 項目, 說明 } ]
    """
    docs = []
    n = len(lines)

    # 找到「二、」且包含「申請程序」的那一點
    start_index = None
    for i, line in enumerate(lines):
        if line.startswith("二、") and "申請程序" in line:
            start_index = i
            break

    if start_index is None:
        return docs  # 找不到就直接回傳空陣列

    # 在第二點裡面往下找到表頭「編號」
    header_index = None
    for i in range(start_index, n):
        if lines[i].strip() == "編號":
            header_index = i
            break

    if header_index is None:
        return docs

    # 依序確認「編號」「項目」「說明」三個表頭（限制搜尋範圍，避免吃到太後面）
    labels = ["編號", "項目", "說明"]
    found = 0
    i = header_index
    MAX_SCAN = 20
    while i < n and found < len(labels) and i < header_index + MAX_SCAN:
        if lines[i].strip() == labels[found]:
            found += 1
        i += 1

    # 若沒找到完整三個欄位，就視為這一版沒有這個表格
    if found < len(labels):
        return docs

    start_rows = i  # 表頭結束後的下一行開始是資料列

    current = None
    item_lines = []
    desc_lines = []

    while start_rows < n:
        line = lines[start_rows].strip()
        start_rows += 1

        if not line:
            continue

        # 遇到下一個大節（例如「三、申請注意事項」）就結束
        if re.match(r'^[一二三四五六七八九十]+、', line):
            break

        # 新的一列：純數字（1、2、3…）
        if re.match(r'^\d+$', line):
            # 先收掉上一筆
            if current is not None:
                current["項目"] = " / ".join(item_lines).strip()
                current["說明"] = "\n".join(desc_lines).strip()
                docs.append(current)

            current = {
                "編號": line,
                "項目": "",
                "說明": ""
            }
            item_lines = []
            desc_lines = []
            continue

        # 還沒遇到第一個編號就略過
        if current is None:
            continue

        # 判斷這一行是「項目」還是「說明」
        is_http_like = line.startswith("http") or line.startswith("https")
        is_link_like = line.startswith("【") or line.startswith("網址")

        # 有句號、逗號等，看起來比較像說明文字
        has_sentence_punct = any(ch in line for ch in "。？！；;:，,")

        # 前幾行、又沒有句子標點、又不是網址/連結 → 視為「項目」
        if (len(item_lines) < 3
                and not is_http_like
                and not is_link_like
                and not has_sentence_punct):
            item_lines.append(line)
        else:
            # 其他都當作說明
            desc_lines.append(line)

    # 迴圈結束，把最後一列收掉
    if current is not None:
        current["項目"] = " / ".join(item_lines).strip()
        current["說明"] = "\n".join(desc_lines).strip()
        docs.append(current)

    return docs


def parse_school_table(lines, start_index):
    """
    解析「編號 / 學校名稱 / 姊妹校要求條件」的表格段落，
    回傳 (schools, next_index)

    每筆為：
    {
        "編號": "1",
        "學校名稱": "日本 大阪學院大學 / Osaka Gakuin University",
        "姊妹校要求條件": "...."
    }
    """
    schools = []
    n = len(lines)

    # 嘗試在 start_index 往下幾行找表頭（第一波有，第二波沒有）
    i = start_index
    headers = ["編號", "學校名稱", "姊妹校要求條件"]
    found = 0
    j = i
    # 限制只往下掃一小段，避免整篇都被吃掉
    while j < n and found < len(headers) and j < i + 20:
        if lines[j].strip() == headers[found]:
            found += 1
        j += 1

    if found == len(headers):
        # 有找到完整三個表頭，就從表頭後面開始讀資料
        i = j
    else:
        # 沒有表頭（例如第二波），直接從 start_index 開始讀（會略過不是數字的那幾行）
        i = start_index

    current = None
    name_lines = []
    req_lines = []
    name_lines_count = 0

    while i < n:
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        # 碰到下一波或下一個大節就停（給呼叫者再處理）
        if re.match(r'^(第一波|第二波)\s*截止日', line):
            i -= 1
            break
        if re.match(r'^[一二三四五六七八九十]+、', line):
            i -= 1
            break

        # 跳過表頭（萬一有又出現一次）
        if line in ("編號", "學校名稱", "姊妹校要求條件"):
            continue

        # 新的一列：純數字
        if re.match(r'^\d+$', line):
            # 收掉上一筆
            if current is not None:
                current["學校名稱"] = " / ".join(name_lines).strip()
                current["姊妹校要求條件"] = "\n".join(req_lines).strip()
                schools.append(current)

            current = {
                "編號": line,
                "學校名稱": "",
                "姊妹校要求條件": ""
            }
            name_lines = []
            req_lines = []
            name_lines_count = 0
            continue

        # 還沒遇到第一個編號就略過
        if current is None:
            continue

        # 判斷「學校名稱」或「姊妹校要求條件」
        is_http_like = line.startswith("http") or line.startswith("https")
        is_link_like = line.startswith("【") or line.startswith("網址")
        is_bracket_only = re.match(r'^\[.*\]$', line) is not None

        # 前 1~2 行、又不是網址／連結 → 當作學校名稱（中 + 英）
        if (name_lines_count < 2
                and not is_http_like
                and not is_link_like
                and not is_bracket_only):
            name_lines.append(line)
            name_lines_count += 1
        else:
            # 其餘全部當作姊妹校要求條件（含網址、備註、條件、條列）
            req_lines.append(line)

    # 收最後一筆
    if current is not None:
        current["學校名稱"] = " / ".join(name_lines).strip()
        current["姊妹校要求條件"] = "\n".join(req_lines).strip()
        schools.append(current)

    return schools, i


def extract_waves(lines):
    """
    解析「第一波 截止日…」「第二波 截止日…」區塊，
    回傳 waves = [
      {
        "wave": "第一波",
        "deadline": "...",
        "schools": [ {編號, 學校名稱, 姊妹校要求條件}, ... ]
      },
      ...
    ]
    """
    waves = []
    n = len(lines)
    i = 0

    wave_pattern = re.compile(r'^(第一波|第二波)\s*截止日[:：]?\s*(.+)$')

    while i < n:
        line = lines[i].strip()
        m = wave_pattern.match(line)
        if m:
            wave_name = m.group(1)       # 第一波 / 第二波
            deadline = m.group(2).strip()

            # 從下一行開始丟給 parse_school_table
            schools, next_i = parse_school_table(lines, i + 1)
            waves.append({
                "wave": wave_name,
                "deadline": deadline,
                "schools": schools
            })
            i = next_i
        else:
            i += 1

    return waves


def crawl_ttu_article(url: str, title: str | None = None) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0 Safari/537.36"
        )
    }

    # 取得網頁（關閉 SSL 憑證驗證）
    resp = requests.get(url, headers=headers, timeout=10, verify=False)
    resp.raise_for_status()

    # 嘗試修正編碼
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")

    # 若沒給 title，就抓網頁 <title>
    if title is None:
        if soup.title:
            title = soup.title.get_text(strip=True)
        else:
            title = ""

    # 取整頁文字
    full_text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]

    # 抓「一、二、三…」標題文字
    section_titles = {}
    section_pattern = re.compile(r'^([一二三四五六七八九十]+)、\s*(.+?)\s*[:：]?$')
    for line in lines:
        m = section_pattern.match(line)
        if m:
            section_titles[m.group(1)] = m.group(2)

    # 第一點：時間（純文字）
    section1_text = ""
    start1 = None
    for i, line in enumerate(lines):
        if line.startswith("一、"):
            start1 = i
            break
    if start1 is not None:
        buf = []
        for j in range(start1 + 1, len(lines)):
            if lines[j].startswith("二、"):
                break
            buf.append(lines[j])
        section1_text = "\n".join(buf).strip()

    # 第二點表格：編號 / 項目 / 說明
    section2_rows = parse_section2_table(lines)

    # 第三點：申請注意事項（純文字）→ 只留到「四、」為止
    section3_note_text = ""
    start3 = None
    for i, line in enumerate(lines):
        if line.startswith("三、"):
            start3 = i
            break
    if start3 is not None:
        buf = []
        for j in range(start3 + 1, len(lines)):
            # 碰到下一個大節「四、」就停
            if lines[j].startswith("四、"):
                break
            buf.append(lines[j])
        section3_note_text = "\n".join(buf).strip()

    # 第四點：承辦人（純文字）
    section4_text = ""
    start4 = None
    for i, line in enumerate(lines):
        if line.startswith("四、"):
            start4 = i
            break
    if start4 is not None:
        buf = []
        for j in range(start4 + 1, len(lines)):
            if lines[j].startswith("五、"):
                break
            buf.append(lines[j])
        section4_text = "\n".join(buf).strip()

    # 第五點：第一波&第二波可申請學校名單（姊妹校列表）
    start5 = None
    for idx, line in enumerate(lines):
        if line.startswith("五、") and ("學校" in line or "名單" in line):
            start5 = idx
            break

    waves = []
    if start5 is not None:
        # 從第五點底下開始解析「第一波 / 第二波」表格
        waves = extract_waves(lines[start5 + 1:])
    else:
        # 若舊版仍放在第三點，就退回全篇掃描（相容舊格式）
        waves = extract_waves(lines)

    result = {
        "title": title,
        "url": url,
        "section1": {
            "title": section_titles.get("一", ""),
            "content": section1_text
        },
        "section2": {
            "title": section_titles.get("二", ""),
            "rows": section2_rows  # 每列：編號 / 項目 / 說明
        },
        # 三、申請注意事項（只放純文字）
        "section3": {
            "title": section_titles.get("三", ""),
            "content": section3_note_text
        },
        # 四、承辦人資訊
        "section4": {
            "title": section_titles.get("四", ""),
            "content": section4_text
        },
        # 五、第一波&第二波可申請學校名單（姊妹校列表）
        "section5": {
            "title": section_titles.get("五", ""),
            "waves": waves  # 每列：編號 / 學校名稱 / 姊妹校要求條件
        }
    }
    return result


def save_to_json(data: dict, filename: str = "ttu_exchange.json") -> None:
    """把爬到的資料存成 JSON 檔"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已輸出 JSON 檔：{filename}")


if __name__ == "__main__":
    url = "https://oia.ttu.edu.tw/p/406-1076-38557,r11.php?Lang=zh-tw"
    data = crawl_ttu_article(
        url,
        title="2026年春季姊妹校交換計畫與雙聯學位申請第一波&第二波資訊"
    )

    # 存成檔案
    save_to_json(data, "ttu_exchange_2026_spring.json")
