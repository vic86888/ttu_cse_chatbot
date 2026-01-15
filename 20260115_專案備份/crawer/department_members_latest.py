# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import json
import os
import urllib.parse
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
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; ttu-dept-members/1.0)"
})
session.mount("https://", NonStrictTLSAdapter())


def main():
    # 定義 JSON 檔案位置(專案根目錄的 data 資料夾)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # 往上一層到專案根目錄
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "department_members.json")
    
    # 初始化資料列表
    members_data = []
    
    source_url = "https://cse.ttu.edu.tw/p/412-1058-157.php?Lang=zh-tw"
    speech_announcement_url = source_url

    # ★ 用寬鬆驗證的 session 來抓
    res = session.get(speech_announcement_url, timeout=10)
    res.raise_for_status()
    # 編碼修正一下，避免亂碼
    if not res.encoding or res.encoding.lower() == "iso-8859-1":
        res.encoding = res.apparent_encoding or "utf-8"

    soup = BeautifulSoup(res.text, 'html.parser')
    seen = set()
    all_info = []
    
    for mdetail in soup.find_all(class_="mdetail"):
        all_info = []
        # print(mdetail.get_text(strip=True))
        name = mdetail.find_all(style="color:#0000cd;")
        n = ', '.join([x.get_text(strip=True) for x in name if x.get_text(strip=True)])
        if n:
            all_info.append(n)
        # print(n)
        
        info = mdetail.find_all('p')
        # print(info)
        for add in info:
            add_str = add.get_text(strip=True)
            
            # 預設沒找到重複
            trash = False
            
            for x in all_info:
                if add_str in x or add_str == '學術研究發表｜研究計畫':
                    trash = True
                    break  # 找到重複就跳出
            if not trash:
                all_info.append(add_str)
        
        if not all_info:
            continue

        # ==========================
        # 兼任老師另外處理
        # ==========================
        if '兼任' in all_info[0]:
            # 拆分姓名和職稱
            full_name = all_info[0].strip()
            parts = full_name.split()
            if len(parts) >= 2:
                name = parts[0]  # 第一部分是姓名
                title = ' '.join(parts[1:])  # 剩餘部分是職稱
            else:
                name = full_name
                title = ''
            
            print("人物:", all_info[0])
            print("姓名:", name)
            print("職稱:", title)
            print("系所:資工系")
            email = ''
            for text in all_info:
                if 'E-mail' in text:
                    email = text
                    print("信箱:"+text)

            # ====== ★ 調整兼任「學歷」與 metadata 的處理 ======
            # 期望：學歷不只限博士，有碩士也要保留整行「學歷：...」
            education_from_metadata = "無"
            metadata_parts = []
            # all_info[1:3] 通常是 [學歷：..., 經歷：...]
            for part in all_info[1:3]:
                if '學歷' in part:
                    # 不再只限「博士」，整行學歷直接收
                    education_from_metadata = part
                else:
                    # 通常就是「經歷：...」，當成 metadata
                    metadata_parts.append(part)
            
            combined_string = "。".join(metadata_parts) if metadata_parts else ""
            print("metadata: " + combined_string)
            print('-'*20)
            
            member = {
                "姓名": name,
                "職稱": title,
                "電話": "無",
                "信箱": email if email else "無",
                "學歷": education_from_metadata,
                "辦公室": "無",
                "metadata": combined_string,
                "資料來源": source_url
            }
            members_data.append(member)
            continue

        # ==========================
        # 專任 / 系辦成員處理
        # ==========================

        # 拆分姓名和職稱
        full_name = all_info[0].strip()
        name = ""
        title = ""
        
        # 處理包含逗號的情況: "姓名 職稱1, 職稱2"
        if ',' in full_name:
            comma_parts = full_name.split(',', 1)
            first_part = comma_parts[0].strip()  # "姓名 職稱1"
            second_part = comma_parts[1].strip()  # "職稱2"
            
            # 拆分第一部分的姓名和職稱
            space_parts = first_part.split(None, 1)
            name = space_parts[0]  # 姓名
            title1 = space_parts[1] if len(space_parts) > 1 else ''
            
            # 合併職稱
            if title1:
                title = f"{title1}, {second_part}"
            else:
                title = second_part
        else:
            # "姓名 職稱"
            space_parts = full_name.split(None, 1)
            if len(space_parts) >= 2:
                name = space_parts[0]
                title = space_parts[1]
            else:
                name = full_name
                title = ''
        
        print("人物:" + all_info[0])
        print("姓名:" + name)
        print("職稱:" + title)
        print("系所:資工系")

        # 確保 index 安全
        phone = all_info[1] if len(all_info) > 1 else "無"
        email = all_info[2] if len(all_info) > 2 else "無"
        office = all_info[3] if len(all_info) > 3 else "無"

        print("電話:" + phone)
        print("信箱:" + email)
        print("辦公室:" + office)
        combined_string = "".join(all_info[4:]) if len(all_info) > 4 else ""
        print("metadata: " + combined_string)

        print('-'*20)
        
        # 先檢查 metadata 中是否包含學歷
        education_from_metadata = "無"
        metadata_clean = combined_string
        if '學歷' in combined_string or '博士' in combined_string or '碩士' in combined_string or '學士' in combined_string:
            import re as _re
            patterns = [
                r'學歷[：:](.*?)(?=專長|經歷|研究|$)',
                r'(.*?博士.*?)(?=專長|經歷|研究|$)',
            ]
            for pattern in patterns:
                match = _re.search(pattern, combined_string, _re.DOTALL)
                if match:
                    edu_text = match.group(1).strip()
                    if '博士' in edu_text:
                        education_from_metadata = edu_text
                        metadata_clean = combined_string.replace(match.group(0), '').strip()
                    break
        
        education = education_from_metadata if education_from_metadata != "無" else "無"

        # 嘗試透過個人頁面抓學歷（博士）
        try:
            link_tag = None
            # 優先找包在外層的 <a>
            parent_a = mdetail.find_parent('a', href=True)
            if parent_a and parent_a.get('href'):
                link_tag = parent_a
            else:
                # 再試著找內部或相鄰的 <a>
                inner_a = mdetail.find('a', href=True)
                if inner_a and inner_a.get('href'):
                    link_tag = inner_a
                else:
                    prev_a = mdetail.find_previous('a', href=True)
                    if prev_a and prev_a.get('href'):
                        link_tag = prev_a

            if link_tag:
                href = link_tag.get('href')
                full_url = urllib.parse.urljoin(source_url, href)

                def fetch_education(page_url):
                    try:
                        r = session.get(page_url, timeout=8)  # ★ 改用 session
                        if not r.encoding or r.encoding.lower() == "iso-8859-1":
                            r.encoding = r.apparent_encoding or "utf-8"
                        sp = BeautifulSoup(r.text, 'html.parser')
                        
                        for element in sp.find_all(['b', 'strong', 'h1', 'h2', 'h3', 'h4']):
                            text = element.get_text(strip=True)
                            if '學歷' in text or 'Education' in text:
                                next_table = element.find_next('table')
                                if next_table:
                                    education_list = []
                                    rows = next_table.find_all('tr')
                                    for row in rows:
                                        cells = row.find_all(['td', 'th'])
                                        if cells:
                                            cell_text = cells[0].get_text(strip=True)
                                            if cell_text and cell_text != '...' and '博士' in cell_text:
                                                education_list.append(cell_text)
                                    if education_list:
                                        return '；'.join(education_list)
                                break
                    except Exception:
                        return "無"
                    return "無"

                if education == "無" and full_url:
                    education = fetch_education(full_url)
        except Exception:
            pass

        member = {
            "姓名": name,
            "職稱": title,
            "系所": "資工系",
            "電話": phone,
            "信箱": email,
            "辦公室": office,
            "學歷": education,
            "metadata": metadata_clean,
            "資料來源": source_url
        }
        members_data.append(member)

    # ==============================
    # 建立輸出 JSON 結構（與你期望的格式一致）
    # ==============================
    output_data = {
        "成員總數": len(members_data),
        "資料來源": source_url,
        "成員列表": members_data
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n成功將 {len(members_data)} 筆資料儲存至 {json_path}")
        


if __name__ == "__main__":
    main()
