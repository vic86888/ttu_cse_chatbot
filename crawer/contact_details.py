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
    json_path = os.path.join(data_dir, "contact_details.json")
    
    # 初始化聯絡資訊列表
    contact_list = []
    
    source_url = "https://recruit.ttu.edu.tw/p/412-1068-2684.php?Lang=zh-tw#start-C"
    speech_announcement_url = source_url

    res = requests.get(speech_announcement_url)
    soup = BeautifulSoup(res.text, 'html.parser')
    all_text1 = []
    all_text2 = []
    department_contacts = []  # 儲存學系聯絡資訊
    
    for table in soup.find_all("table"):
        # 爬取「項目」相關表格
        data = table.find_all('th',string = '項目') 
        if data:
            td = table.find_all('td')
            if td:
                for text in td:
                    all_text1.append(text.get_text(strip=True))

        # 爬取「業務項目」相關表格
        data = table.find_all('th',string = '業務項目')
        if data:
            td = table.find_all('td')
            if td:
                for text in td:
                    all_text2.append(text.get_text(strip=True))
        
        # 爬取「學系」相關表格
        th_list = table.find_all('th')
        if any('學系' in th.get_text(strip=True) for th in th_list):
            # 檢查是否為學系聯絡表格(包含「聯絡人員」標題)
            if any('聯絡人員' in th.get_text(strip=True) for th in th_list):
                rows = table.find_all('tr')
                for row in rows[1:]:  # 跳過標題列
                    cells = row.find_all('td')
                    if cells:
                        # 處理每一列的資料
                        cell_texts = [cell.get_text(strip=True) for cell in cells]
                        # 過濾掉空白儲存格
                        cell_texts = [text for text in cell_texts if text]
                        department_contacts.extend(cell_texts)

    for i in range(0, len(all_text1), 3):  # 步進3
        item = all_text1[i]            # 事項
        people = all_text1[i + 1]      # 負責人
        num = all_text1[i + 2]         # 電話
        full_phone = f"(02)2182-2928 分機 {num}"
        print(f"事項: {item}, 負責人: {people},  聯絡電話: {full_phone}")
        
        # 建立聯絡資訊字典
        contact_info = {
            "辦理項目": item,
            "承辦人": people,
            "分機": full_phone,
            "資料來源": source_url
        }
        
        # 檢查是否已存在相同項目,若存在則更新,否則新增
        existing = next((c for c in contact_list if c.get("辦理項目") == item), None)
        if existing:
            existing.update(contact_info)
        else:
            contact_list.append(contact_info)

    for i in range(0, len(all_text2), 3):  # 步進3
        people = all_text2[i]            
        item = all_text2[i + 1]      
        num = all_text2[i + 2]         # 電話
        full_phone = f"(02)2182-2928 分機 {num}"
        print(f"事項: {item}, 負責人: {people},  聯絡電話: {full_phone}")
        
        # 建立聯絡資訊字典
        contact_info = {
            "辦理項目": item,
            "承辦人": people,
            "分機": full_phone,
            "資料來源": source_url
        }
        
        # 檢查是否已存在相同項目,若存在則更新,否則新增
        existing = next((c for c in contact_list if c.get("辦理項目") == item), None)
        if existing:
            existing.update(contact_info)
        else:
            contact_list.append(contact_info)

    # 處理學系聯絡資訊
    print("\n" + "="*50)
    print("處理學系聯絡資訊")
    print("="*50)
    
    i = 0
    current_department = None
    
    while i < len(department_contacts):
        text = department_contacts[i]
        
        # 判斷是否為學系名稱(包含「學系」或「學士班」)
        if '學系' in text or '學士班' in text:
            current_department = text
            i += 1
            continue
        
        # 判斷是否為聯絡人員(包含「主任」或「小姐」或「先生」)
        if current_department and ('主任' in text or '小姐' in text or '先生' in text):
            person = text
            # 檢查下一個是否為分機號碼
            if i + 1 < len(department_contacts):
                next_text = department_contacts[i + 1]
                # 如果下一個是純數字,則視為分機
                if next_text.isdigit():
                    extension = next_text
                    full_phone = f"(02)2182-2928 分機 {extension}"
                    print(f"學系: {current_department}, 聯絡人員: {person}, 分機: {full_phone}")
                    
                    # 建立學系聯絡資訊字典
                    dept_info = {
                        "學系": current_department,
                        "聯絡人員": person,
                        "分機": full_phone,
                        "資料來源": source_url
                    }
                    
                    # 加入聯絡資訊列表
                    contact_list.append(dept_info)
                    i += 2  # 跳過聯絡人員和分機
                    continue
        
        i += 1

    # 將資料寫入 JSON 檔案
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(contact_list, f, ensure_ascii=False, indent=2)
        print("\n" + "="*50)
        print(f"成功將 {len(contact_list)} 筆聯絡資訊儲存至 {json_path}")
        print("="*50)
    except Exception as e:
        print(f"寫入 JSON 檔案失敗: {e}")
        
if __name__ == "__main__":
    main()