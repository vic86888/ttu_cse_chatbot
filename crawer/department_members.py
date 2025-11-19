import requests
from bs4 import BeautifulSoup
import json
import os

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
            print('-'*20)
            
            # 新增到資料列表
            member = {
                "姓名": name,
                "職稱": title,
                "電話": "無",
                "信箱": email if email else "無",
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
        
        # 新增到資料列表
        member = {
            "姓名": name,
            "職稱": title,
            "系所":"資工系",
            "電話": all_info[1],
            "信箱": all_info[2],
            "辦公室": all_info[3],
            "metadata": combined_string,
            "資料來源": source_url
        }
        members_data.append(member)

    # 將資料寫入 JSON 檔案
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(members_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n成功將 {len(members_data)} 筆資料儲存至 {json_path}")
        
if __name__ == "__main__":
    main()