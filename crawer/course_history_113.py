import requests
from bs4 import BeautifulSoup
import json
import os

def get_course_history(school_year='113', semester='1', department='06', class_level='U0', grade_name=''):
    """
    爬取大同大學資工系歷史課程查詢網站的課程資訊
    
    參數:
        school_year (str): 學年度,例如 '113'
        semester (str): 學期 '1': 第一學期(上), '2': 第二學期(下)
        department (str): 系所代碼
            '01': 機材系所, '02': 電機系所, '03': 化生系所
            '04': 工設系所, '05': 事經系所, '06': 資工系所
            '12': 資經系所, '14': 媒設系, '15': 應外系
            '17': 設科所, '19': 工程學院學士班
        class_level (str): 班級代碼
            'U0': 大學部全部, 'U1': 大學部一年級, 'U2': 大學部二年級
            'U3': 大學部三年級, 'U4': 大學部四年級
            'G0': 碩士班, 'E0': 碩士在職專班, 'D0': 博士班
        grade_name (str): 年級名稱,如 '一年級'、'二年級'等
    
    返回:
        list: 課程資料列表
    """
    url = "https://tchinfo.ttu.edu.tw/couquery/historysbj.php"
    
    # 設定 headers 模擬瀏覽器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://tchinfo.ttu.edu.tw',
        'Referer': 'https://tchinfo.ttu.edu.tw/couquery/historysbj.php'
    }
    
    # 建立 session 以保持連線
    session = requests.Session()
    
    # 首先訪問網頁以獲取初始的 cookies 和 session
    print("正在訪問網頁...")
    response = session.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"無法訪問網頁,狀態碼: {response.status_code}")
        return None
    
    # 準備表單資料
    form_data = {
        'SchYear': school_year,
        'Sem': semester,
        'SelDp': department,
        'SelCl': class_level
    }
    
    # 建立學年學期標籤
    semester_label = f"{school_year}{'上' if semester == '1' else '下'}"
    
    print(f"正在提交表單 (學年度: {school_year}, 學期: {semester_label}, 系所代碼: {department}, 班級: {class_level})...")
    # 提交表單
    response = session.post(url, data=form_data, headers=headers)
    
    if response.status_code != 200:
        print(f"表單提交失敗,狀態碼: {response.status_code}")
        return None
    
    # 解析 HTML
    print("正在解析資料...")
    response.encoding = 'utf-8'  # 確保正確的編碼
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 尋找所有的 table
    tables = soup.find_all('table')
    
    if not tables:
        print("未找到表格資料")
        # 儲存 HTML 以便檢查
        with open('debug_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("已將回應儲存至 debug_response.html 供檢查")
        return None
    
    courses = []
    
    # 尋找包含課程資料的表格
    for table in tables:
        rows = table.find_all('tr')
        
        # 跳過空表格
        if len(rows) <= 1:
            continue
        
        # 檢查是否為課程表格 (通常第一行會有表頭)
        headers = []
        first_row = rows[0]
        th_cells = first_row.find_all(['th', 'td'])
        
        if not th_cells:
            continue
            
        # 取得表頭
        headers = [cell.get_text(strip=True) for cell in th_cells]
        
        # 解析資料行
        for row in rows[1:]:
            cells = row.find_all('td')
            
            if len(cells) == 0:
                continue
            
            # 建立課程字典,只取前6個欄位:序號、課號、課程名稱、教師、選別、學分
            course = {
                '學年學期': semester_label,
                '所屬年級': grade_name if grade_name else '全年級',
                '資料來源': url
            }
            for i, cell in enumerate(cells):
                if i>0 and i < 6 and i < len(headers):  # 只取前6個欄位
                    course[headers[i]] = cell.get_text(strip=True)
            
            # 檢查課程名稱,過濾掉"專任教師合計"和"兼任教師合計"
            if course and '課程名稱' in course:
                course_name = course['課程名稱']
                if course_name not in ['專任教師合計', '兼任教師合計']:
                    courses.append(course)
    
    print(f"成功爬取 {len(courses)} 筆課程資料")
    
    return courses


if __name__ == "__main__":
    print("=== 大同大學113學年度課程資料爬蟲 ===\n")
    
    # 定義要爬取的年級
    grade_levels = [
        ('U1', '一年級'),
        ('U2', '二年級'),
        ('U3', '三年級'),
        ('U4', '四年級')
    ]
    
    # 定義學期
    semesters = [
        ('1', '第一學期'),
        ('2', '第二學期')
    ]
    
    all_courses = []
    
    # 逐一爬取每個學期和年級的課程
    for semester_num, semester_name in semesters:
        print(f"\n{'='*50}")
        print(f"開始爬取 113學年度{semester_name}")
        print('='*50)
        
        for class_level, grade_name in grade_levels:
            print(f"\n正在爬取 {grade_name} 課程...")
            courses = get_course_history(
                school_year='113', 
                semester=semester_num, 
                department='06', 
                class_level=class_level,
                grade_name=grade_name
            )
            
            if courses:
                print(f"  ✓ {grade_name} 爬取成功: {len(courses)} 筆課程")
                all_courses.extend(courses)
            else:
                print(f"  ✗ {grade_name} 爬取失敗")
    
    if all_courses:
        # 儲存到專案根目錄的 data 資料夾
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_file = os.path.join(data_dir, 'course_history_113.json')
        
        # 建立包含所有資料的總字典結構
        # 總覽包含課程列表，課程列表是所有課程的 dictionary 陣列
        output_data = {
            "總覽": {
                "課程總數": len(all_courses),
                "資料來源": "https://tchinfo.ttu.edu.tw/couquery/historysbj.php",
                "課程列表": all_courses
            }
        }
        
        # 將每個課程作為獨立的 dictionary 加入
        for idx, course in enumerate(all_courses, start=1):
            output_data[f"課程{idx}"] = course
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*50}")
        print(f"課程資料已儲存至 {output_file}")
        print(f"總計爬取 {len(all_courses)} 筆課程資料")
        print(f"{'='*50}")
        
        # 統計每個學期和年級的課程數量
        print("\n學期統計:")
        from collections import Counter
        semester_count = Counter(course['學年學期'] for course in all_courses)
        for sem, count in sorted(semester_count.items()):
            print(f"  {sem}: {count} 筆")
        
        print("\n年級統計:")
        grade_count = Counter(course['所屬年級'] for course in all_courses)
        for grade, count in sorted(grade_count.items()):
            print(f"  {grade}: {count} 筆")
        
        if len(all_courses) > 0:
            print("\n前三筆課程資料範例:")
            for i, course in enumerate(all_courses[:3], 1):
                print(f"\n第 {i} 筆:")
                print(json.dumps(course, ensure_ascii=False, indent=2))
    else:
        print("\n爬取失敗,請檢查 debug_response.html 以了解問題")
