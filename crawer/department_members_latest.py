import requests
from bs4 import BeautifulSoup
import json
import os
import urllib.parse

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

    res = requests.get(speech_announcement_url)
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
        
        if '兼任' in all_info[0]:
            # 拆分姓名和職稱
            full_name = all_info[0].strip()
            # 嘗試使用空格分割
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
            combined_string = "。".join(all_info[1:3])
            print("metadata: " + combined_string)
            
            # 檢查 metadata 中是否包含學歷，若有則提取
            education_from_metadata = "無"
            metadata_parts = []
            for part in all_info[1:3]:
                if '學歷' in part or '博士' in part or '碩士' in part or '學士' in part:
                    # 只保留博士學歷
                    if '博士' in part:
                        education_from_metadata = part
                else:
                    metadata_parts.append(part)
            
            # 重新組合不含學歷的 metadata
            combined_string = "。".join(metadata_parts) if metadata_parts else ""
            
            print('-'*20)
            
            # 新增到資料列表
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
        # 處理有空格的情況: "姓名 職稱"
        else:
            # 使用split(None, 1)會自動處理所有類型的空白字元
            space_parts = full_name.split(None, 1)
            if len(space_parts) >= 2:
                name = space_parts[0]
                title = space_parts[1]
            else:
                # 只有姓名,沒有空格
                name = full_name
                title = ''
        
        print("人物:" + all_info[0])
        print("姓名:" + name)
        print("職稱:" + title)
        print("系所:資工系")
        print("電話:" + all_info[1])
        print("信箱:" + all_info[2])
        print("辦公室:" + all_info[3])
        combined_string = "".join(all_info[4:])  # 把從癔4個開始的字串合併
        print("metadata: " + combined_string)

        print('-'*20)
        
        # 先檢查 metadata 中是否包含學歷
        education_from_metadata = "無"
        metadata_clean = combined_string
        if '學歷' in combined_string or '博士' in combined_string or '碩士' in combined_string or '學士' in combined_string:
            # 嘗試分離學歷資訊
            import re
            # 尋找學歷相關的段落
            patterns = [
                r'學歷[：:](.*?)(?=專長|經歷|研究|$)',
                r'(.*?博士.*?)(?=專長|經歷|研究|$)',
            ]
            for pattern in patterns:
                match = re.search(pattern, combined_string, re.DOTALL)
                if match:
                    edu_text = match.group(1).strip()
                    # 只保留博士學歷
                    if '博士' in edu_text:
                        education_from_metadata = edu_text
                        # 從 metadata 中移除學歷部分
                        metadata_clean = combined_string.replace(match.group(0), '').strip()
                    break
        
        # 嘗試尋找與此 mdetail 相關的連結 (父節點或內部的 <a>)，以抓取學歷
        education = education_from_metadata if education_from_metadata != "無" else "無"
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
                # 只在非空且看起來是本站或可用的 URL 時請求
                if full_url:
                    def fetch_education(page_url):
                        try:
                            r = requests.get(page_url, timeout=8)
                            r.encoding = r.apparent_encoding  # 使用與主程式一致的編碼偵測方式
                            sp = BeautifulSoup(r.text, 'html.parser')
                            
                            # 尋找包含「學歷」或「Education」的 <b> 或 <strong> 標籤
                            for element in sp.find_all(['b', 'strong', 'h1', 'h2', 'h3', 'h4']):
                                text = element.get_text(strip=True)
                                if '學歷' in text or 'Education' in text:
                                    # 找下一個表格
                                    next_table = element.find_next('table')
                                    if next_table:
                                        education_list = []
                                        rows = next_table.find_all('tr')
                                        for row in rows:
                                            cells = row.find_all(['td', 'th'])
                                            if cells:
                                                cell_text = cells[0].get_text(strip=True)
                                                # 只保留包含「博士」的學歷
                                                if cell_text and cell_text != '...' and '博士' in cell_text:
                                                    education_list.append(cell_text)
                                        if education_list:
                                            return '；'.join(education_list)
                                    break
                        except Exception:
                            return "無"
                        return "無"

                    # 只在還沒有學歷時才去抓取
                    if education == "無":
                        education = fetch_education(full_url)
        except Exception:
            pass

        # 新增到資料列表
        member = {
            "姓名": name,
            "職稱": title,
            "系所":"資工系",
            "電話": all_info[1],
            "信箱": all_info[2],
            "辦公室": all_info[3],
            "學歷": education,
            "metadata": metadata_clean,
            "資料來源": source_url
        }
        members_data.append(member)

    # 建立包含所有資料的總字典結構
    # 總覽包含成員列表，成員列表是所有成員的 dictionary 陣列
    output_data = {
        "總覽": {
            "成員總數": len(members_data),
            "資料來源": source_url,
            "成員列表": members_data
        }
    }
    
    # 將每個成員作為獨立的 dictionary 加入
    for idx, member in enumerate(members_data, start=1):
        output_data[f"成員{idx}"] = member
    
    # 將資料寫入 JSON 檔案
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n成功將 {len(members_data)} 筆資料儲存至 {json_path}")
        
if __name__ == "__main__":
    main()