from pathlib import Path
from paddleocr import PPStructureV3
import pandas as pd
from io import StringIO

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import re
from opencc import OpenCC

# ---------- 0. 建立簡體 -> 繁體 轉換器 ----------
cc = OpenCC("s2tw")  # 簡體轉台灣繁體


def to_trad(text):
    if isinstance(text, str):
        return cc.convert(text)
    return text


def fix_ocr_course_code(code: str) -> str:
    """
    修正 OCR 把大寫 I 誤辨識成數字 1 的情況。
    """
    if code is None:
        return ""

    code = str(code).strip()

    # 把 'nan' 也當成沒有代碼
    if code.lower() == "nan" or code == "":
        return ""

    # 如果本來就是「英文 + 3~4 位數字」，直接視為正常代碼
    # 例如：E1050, G121, N1260, W553, V5030 ...
    if re.fullmatch(r"[A-Z][0-9]{3,4}", code):
        return code

    # 1 + 3 或 4 位數字，全部改成 I + 後面那串
    if re.fullmatch(r"1[0-9]{3,4}", code):
        return "I" + code[1:]

    # 其他情況維持原樣
    return code


# ---------- 1. 初始化 PaddleOCR 表格結構模型 ----------
pipeline = PPStructureV3(
    lang="chinese_cht",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)


# ---------- 2. 一次設定多個要爬的頁面 ----------
sites = [
    {
        "name": "site1",  # AI 智慧學程
        "url": "https://cse.ttu.edu.tw/p/405-1058-11495,c2021.php",
        "selector": "#Dyn_2_2 > div.module.module-detail.md_style1 > div > "
                    "section > div.mcont > div > div > p > img",
    },
    {
        "name": "site2",  # 物聯網學程
        "url": "https://cse.ttu.edu.tw/p/405-1058-11494,c2021.php",
        "selector": "#Dyn_2_2 > div.module.module-detail.md_style1 > div > "
                    "section > div.mcont > div > div > p > img",
    },
    {
        "name": "site3",  # 資通安全學程
        "url": "https://cse.ttu.edu.tw/p/405-1058-11493,c2021.php",
        "selector": "#Dyn_2_2 > div.module.module-detail.md_style1 > div > "
                    "section > div.mcont > div > div > p > img",
    },
    {
        "name": "site4",  # 資通安全與數位鑑識學程
        "url": "https://cse.ttu.edu.tw/p/405-1058-28128,c2021.php",
        "selector": "#Dyn_2_2 > div.module.module-detail.md_style1 > div > "
                    "section > div.mcont > div > div > p > img",
    },
]

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0 Safari/537.36"
    )
}

# 儲存表格圖片
img_dir = Path("tables")
img_dir.mkdir(exist_ok=True)

img_paths = []

# ★ 記錄「圖片檔名（不含副檔名）」對應的原始頁面網址
img_src_map = {}

# ---------- 2-1. 爬各站的表格圖片 ----------
for site in sites:
    print(f"=== 爬取網站：{site['name']} ===")
    resp = requests.get(site["url"], headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # 只抓該網站指定 selector 底下的 img
    img_tags = soup.select(site["selector"])

    for img in img_tags:
        src = img.get("src")
        if not src:
            continue

        # 只抓常見圖片格式（通常這裡就是你的表格圖）
        if not any(src.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]):
            continue

        img_url = urljoin(site["url"], src)

        # 清掉 ? 參數，只取副檔名
        ext = Path(img_url.split("?")[0]).suffix or ".png"
        # 檔名前面加 site 名稱避免撞名
        filename = img_dir / f"{site['name']}_table_{len(img_paths) + 1}{ext}"

        print("downloading:", img_url, "->", filename)
        img_resp = requests.get(img_url, headers=headers)
        img_resp.raise_for_status()
        filename.write_bytes(img_resp.content)

        img_paths.append(filename)

        # ★ 把「圖片檔名（不含副檔名）」對應到「此網站網址」
        img_src_map[filename.stem] = site["url"]

print(f"共下載 {len(img_paths)} 張圖片（來自多個網站）")

# ---------- 3. 丟進 PPStructureV3 產生 markdown ----------
output_dir = Path("output_ppstruct")
output_dir.mkdir(exist_ok=True)

for img_path in img_paths:
    print("processing:", img_path)
    output = pipeline.predict(str(img_path))

    for res in output:
        # 只存 markdown，後面用 read_html 解析
        res.save_to_markdown(str(output_dir))

# ---------- 4. 從 markdown 把文字 + <table> 抓出來轉成 JSON ----------
# 一張 md（=一張圖片）輸出一個 JSON 檔（= 一張圖片一個 JSON）

for md_path in output_dir.glob("*.md"):
    print("parsing markdown:", md_path)
    text = md_path.read_text(encoding="utf-8")

    # 4-0 整份 markdown 先轉成台灣繁體
    text = to_trad(text)

    # ★ 從 md 檔名反查對應的圖片檔名，進而找到原始網址
    #   save_to_markdown 通常會用「圖片檔名 + 後綴」命名，例如 xxx_structure.md
    md_stem = md_path.stem  # 不含 .md
    candidate_stems = [
        md_stem,
        re.sub(r"_structure$", "", md_stem),
    ]

    source_url = ""
    for s in candidate_stems:
        if s in img_src_map:
            source_url = img_src_map[s]
            break

    # 對「這一張圖片」的所有課程紀錄
    records = []

    # 4-1 抓 title（第一個非空行），並去掉前面的 markdown 標題符號 ##
    title = ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            # 把開頭的 # 或 ## 之類的 markdown 標題符號拿掉
            title = re.sub(r'^#+\s*', '', line)
            break

    # 4-2 從全文抓「設置宗旨」、「適用對象」
    purpose_text = ""
    target_text = ""

    m = re.search(r"[二2]、\s*設置宗旨[:：]\s*(.+?)[三3]、", text, flags=re.S)
    if m:
        purpose_text = m.group(1).strip()

    m = re.search(r"[三3]、\s*適用對象[:：]\s*(.+?)[四4]、", text, flags=re.S)
    if m:
        target_text = m.group(1).strip()

    # 4-3 抓表格（此時表格內容也已經是繁體）
    try:
        tables = pd.read_html(StringIO(text))
    except ValueError:
        # 沒 table 就跳過這個 md
        continue

    for df in tables:
        # -------- 4-3-1. 找 header 那一列（含「課程代碼」、「課程名稱」） --------
        header_idx = None
        # 不只看前幾列，整張表都掃一遍比較保險
        for idx in range(len(df)):
            row_text = "".join(str(v) for v in df.iloc[idx].tolist())
            if "課程代碼" in row_text and "課程名稱" in row_text:
                header_idx = idx
                break

        # 沒有找到 header，就跳過這張表
        if header_idx is None:
            continue

        # 不要把 header 之前的列砍掉（那些常常是「一、核心課程」這種類別標題）
        header = [str(v).strip() for v in df.iloc[header_idx]]
        df.columns = header
        # 只把 header 那一列本身去掉
        df = df.drop(index=header_idx).reset_index(drop=True)

        current_category = ""  # 當前課程類別（基礎 / 核心 / 進階 / 實務 / 進階與實務）

        for _, row in df.iterrows():
            # 把這一列所有欄位接起來當成一段文字來判斷
            row_values = [str(v) for v in row.tolist()]
            row_text = "".join(row_values)

            raw_code = str(row.get("課程代碼", "")).strip()
            raw_name = str(row.get("課程名稱", "")).strip()
            raw_credit = str(row.get("學分數", "")).strip()

            # -------- 4-3-2. 判斷是不是「類別標題列」 --------
            # 特色：
            #   * 像「二、進階課程(至少9學分)」這種，code/name/credit 通常三個都一樣
            #   * 或是「xxx課程」之類、而且沒有任何英數字（不像真正課程代碼）
            is_category_like_row = (
                (raw_code and raw_code == raw_name == raw_credit)
                or ("課程" in raw_code and not re.search(r"[A-Za-z0-9]", raw_code))
            )

            if is_category_like_row:
                # 從這一列裡面抓出真正的「xxx課程」字樣
                m_cat = re.search(
                    r"(基礎課程|核心課程|[進追]階與實務課程|進階課程|實務課程)",
                    row_text,
                )
                if m_cat:
                    key = m_cat.group(1)
                    # 「進階」被 OCR 成「追階」也一起吃掉
                    if "階與實務課程" in key:
                        current_category = "進階與實務課程"
                    else:
                        # 直接用「基礎課程 / 核心課程 / 進階課程 / 實務課程」
                        current_category = key
                # 這一列只是分類標題，不要產生課程紀錄
                continue

            # 再保險一次：有些類別列不一定三欄都一樣，用全文再掃一次關鍵字
            m_cat2 = re.search(
                r"(基礎課程|核心課程|[進追]階與實務課程|進階課程|實務課程)",
                row_text,
            )
            if m_cat2:
                key = m_cat2.group(1)
                if "階與實務課程" in key:
                    current_category = "進階與實務課程"
                else:
                    current_category = key
                continue

            # -------- 4-3-3. 真正的課程列，開始取欄位 --------
            code = to_trad(raw_code)
            code = fix_ocr_course_code(code)  # 在這裡做 I / 1 修正

            name = to_trad(raw_name)
            credit = raw_credit
            note = to_trad(str(row.get("備註", "")).strip())

            # 跳過空列或再次出現的 header
            if (not code) or code in ["課程代碼", "代碼"]:
                continue

            record = {
                "title": to_trad(title),
                "設置宗旨": to_trad(purpose_text),
                "適用對象": to_trad(target_text),
                "課程類別": current_category,
                "課程代碼": code,
                "課程名稱": name,
                "學分數": credit,
                "備註": note,
                "資料來源": source_url,
            }
            records.append(record)

    # 4-4 這張圖片有抓到課程才寫 JSON（= 一張圖片一個 JSON）
    if records:
        json_path = md_path.with_suffix(".json")  # 同名 .json 檔
        json_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("saved:", json_path)
