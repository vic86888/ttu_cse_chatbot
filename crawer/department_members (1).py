import requests
from bs4 import BeautifulSoup
import json, os, sys
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from requests.exceptions import SSLError

SPEECH_URL = "https://cse.ttu.edu.tw/p/412-1058-157.php?Lang=zh-tw"

def get_html(url: str) -> str:
    # 用 session 加 retry，比較耐網路抖動
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    retries = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))

    try:
        # 先嘗試正常（驗證 SSL）
        r = s.get(url, timeout=15)
        r.raise_for_status()
        return r.text
    except SSLError as e:
        # 憑證問題 → 降級關閉驗證，只針對這個站！
        sys.stderr.write(f"[warn] SSL 驗證失敗，改用 verify=False 重新嘗試：{e}\n")
        r = s.get(url, timeout=15, verify=False)
        r.raise_for_status()
        return r.text

def parse_members(html: str):
    soup = BeautifulSoup(html, "html.parser")
    members_data = []
    for mdetail in soup.find_all(class_="mdetail"):
        all_info = []
        name_tags = mdetail.find_all(style="color:#0000cd;")
        n = ", ".join([x.get_text(strip=True) for x in name_tags if x.get_text(strip=True)])
        if n:
            all_info.append(n)

        info = mdetail.find_all("p")
        for add in info:
            add_str = add.get_text(strip=True)

            trash = False
            for x in all_info:
                if add_str in x or add_str == "學術研究發表｜研究計畫":
                    trash = True
                    break
            if not trash and add_str:
                all_info.append(add_str)

        if not all_info:
            continue

        def safe(idx, default="無"):
            return all_info[idx] if len(all_info) > idx and all_info[idx] else default

        if "兼任" in all_info[0]:
            email = ""
            for text in all_info:
                if "E-mail" in text or "E-mail:" in text or "Email" in text:
                    email = text
                    break
            combined_string = "。".join(all_info[1:3])
            member = {
                "人物": all_info[0],
                "電話": "無",
                "信箱": email if email else "無",
                "辦公室": "無",
                "metadata": combined_string
            }
        else:
            member = {
                "人物": safe(0, "無"),
                "電話": safe(1, "無"),
                "信箱": safe(2, "無"),
                "辦公室": safe(3, "無"),
                "metadata": "".join(all_info[4:]) if len(all_info) > 4 else ""
            }

        members_data.append(member)

    return members_data

def main():
    json_path = os.path.join(os.getcwd(), "department_members.json")
    html = get_html(SPEECH_URL)
    members = parse_members(html)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(members, f, ensure_ascii=False, indent=4)

    print(f"成功將 {len(members)} 筆資料儲存至 {json_path}")

if __name__ == "__main__":
    main()
