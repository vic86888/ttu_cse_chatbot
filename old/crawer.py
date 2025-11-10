import requests
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin, urlparse
from pathlib import Path


url = "https://cse.ttu.edu.tw/p/412-1058-2021.php"  # 換成實際網址
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 建議加上逾時與重試（簡版）
resp = requests.get(url, headers=headers, timeout=15)
resp.raise_for_status()  # 非 2xx 會丟例外

# 若網站未正確宣告編碼，可手動設定（必要時）
# resp.encoding = "utf-8"

soup = BeautifulSoup(resp.text, "lxml")  # 也可用 "html.parser"

classify = soup.find("div", class_="module module-path md_style1")

if classify:
    text = classify.get_text(strip=True, separator=" / ")
    print(text)
    # 如要另存檔
    # with open("module module-path md_style1.html", "w", encoding="utf-8") as f:
    #     f.write(pretty_html)
else:
    print("找不到 <div class='module module-path md_style1'>")

print("\n" + "="*50 + "\n")

title = soup.find("title")
if title:
    # 使用 prettify 美化排版
    text = title.get_text(strip=True)

    print(text)
    # # 如要另存檔
    # with open("title.html", "w", encoding="utf-8") as f:
    #     f.write(pretty_html)
else:
    print("找不到 <div class='title'>")

print("\n" + "="*50 + "\n")

# 取第一個符合的 div
box = soup.find("div", class_="module module-cglist md_style1")
box1 = soup.find("div", class_="module module-detail md_style1")
box2 = soup.find("div", class_="module module-special md_style1")
if box:
    # 使用 prettify 美化排版
    pretty_html = box.prettify()

    # print(pretty_html)
    # # 如要另存檔
    # with open("box.html", "w", encoding="utf-8") as f:
    #     f.write(pretty_html)
if box1:
    # 使用 prettify 美化排版
    pretty_html = box1.prettify()

    # print(pretty_html)
    # # 如要另存檔
    # with open("box1.html", "w", encoding="utf-8") as f:
    #     f.write(pretty_html)
if box2:
    # 使用 prettify 美化排版
    pretty_html = box2.prettify()

    # print(pretty_html)
    # # 如要另存檔
    # with open("box2.html", "w", encoding="utf-8") as f:
    #     f.write(pretty_html)

# 合併多個 Tag
tags = [box, box1, box2]
tags = [t for t in tags if t is not None]  # 過濾 None

if tags:
    main = tags[0]  # 選第一個作為主 Tag
    for t in tags[1:]:
        main.append(t)  # 把其他 Tag 結果 append 到主 Tag 裡
    
    # 現在 main 就是一個新的 Tag，內含所有內容
    # print(main.prettify())

    # with open("main.html", "w", encoding="utf-8") as f:
    #      f.write(main.prettify())

    def extract_stream_in_order(main, base_url, timeout=10):
        """
        依 DOM 順序（document order）走訪 main，將遇到的項目依序轉成事件：
        - {"type": "pdf"|"word"|"image"|"announcement"|"text", "value": ..., "node": tag, "index": i}
        其中 text 會只收「可讀文字」（過濾掉純空白）。
        """
        stream = []
        seen_urls = set()  # 避免重複（例如 <a><img></a> 同時指向同一張圖）

        def classify_link(href_abs):
            # 副檔名快速分類
            ext = Path(urlparse(href_abs).path or "").suffix.lower()
            if ext == ".pdf":
                return "pdf"
            if ext in {".doc", ".docx"}:
                return "word"
            if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}:
                return "image"
            # 無副檔名或 .php/.aspx → HEAD 看 Content-Type
            try:
                r = requests.head(href_abs, allow_redirects=True, timeout=timeout)
                ctype = (r.headers.get("Content-Type") or "").lower()
            except requests.RequestException:
                ctype = ""
            if ctype.startswith("application/pdf"):
                return "pdf"
            if ctype.startswith(("application/msword",
                                "application/vnd.openxmlformats-officedocument.wordprocessingml")):
                return "word"
            if ctype.startswith("image/"):
                return "image"
            return "announcement"

        i = 0
        for node in main.descendants:  # 依文件順序走訪
            # 1) 純文字節點（過濾空白）
            if isinstance(node, NavigableString):
                txt = str(node).strip()
                if txt:
                    stream.append({"type": "text", "value": txt, "node": node, "index": i})
                    i += 1
                continue

            # 2) <img>：視為 image 事件（即使不包在 <a> 裡也會記錄）
            if getattr(node, "name", None) == "img" and node.get("src"):
                src_abs = urljoin(base_url, node["src"])
                if src_abs not in seen_urls:
                    seen_urls.add(src_abs)
                    stream.append({"type": "image", "value": src_abs, "node": node, "index": i})
                    i += 1
                continue

            # 3) <a href=...>：依連結目標分類（pdf/word/image/announcement）
            if getattr(node, "name", None) == "a" and node.has_attr("href"):
                href_abs = urljoin(base_url, node["href"].strip())
                if href_abs not in seen_urls:
                    seen_urls.add(href_abs)
                    kind = classify_link(href_abs)
                    stream.append({"type": kind, "value": href_abs, "node": node, "index": i})
                    i += 1

        return stream

    # ====== 用法（接上你合併 main 的程式之後） ======
    stream = extract_stream_in_order(main, base_url=url)

    # 1) 依原順序輸出
    for ev in stream:
        print(f"[{ev['index']:04d}] {ev['type']}: {ev['value']}")

    # 2) 如果你仍想要分類後的清單，但維持原順序，可用穩定排序或直接過濾：
    pdfs  = [ev['value'] for ev in stream if ev['type'] == 'pdf']
    words = [ev['value'] for ev in stream if ev['type'] == 'word']
    imgs  = [ev['value'] for ev in stream if ev['type'] == 'image']
    texts = [ev['value'] for ev in stream if ev['type'] == 'text']

else:
    print("找不到任何指定的 <div> 標籤")