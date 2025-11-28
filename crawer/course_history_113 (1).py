# -*- coding: utf-8 -*-
import os
import json
import ssl
from collections import Counter

import requests
from bs4 import BeautifulSoup
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
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://tchinfo.ttu.edu.tw',
        'Referer': 'https://tchinfo.ttu.edu.tw/couquery/historysbj.php',
    }

    # 建立學年學期標籤
    semester_label = f"{school_year}{'上' if semester == '1' else '下'}"

    # 先訪問網頁取得 cookies / session
    print("正在訪問網頁...")
    try:
        response = session.get(url, headers=headers, timeout=10)
    except requests.exceptions.SSLError as e:
        print(f"SSL 連線錯誤: {e}")
        return None

    if response.status_code != 200:
        print(f"無法訪問網頁, 狀態碼: {response.status_code}")
        return None

    # 準備表單資料
    form_data = {
        'SchYear': school_year,
        'Sem': semester,
        'SelDp': department,
        'SelCl': class_level,
    }

    print(f"正在提交表單 (學年度: {school_year}, 學期: {semester_label}, 系所代碼: {department}, 班級: {class_level})...")
    try:
        response = session.post(url, data=form_data, headers=headers, timeout=10)
    except requests.exceptions.SSLError as e:
        print(f"SSL 連線錯誤(POST): {e}")
        return None

    if response.status_code != 200:
        print(f"表單提交失敗, 狀態碼: {response.status_code}")
        return None

    # 嘗試用正確編碼解析（避免 Big5 / UTF-8 亂碼）
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"

    print("正在解析資料...")
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    # 尋找所有的 table
    tables = soup.find_all('table')

    if not tables:
        print("未找到表格資料")
        # 儲存 HTML 以便檢查
        with open('debug_response.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print("已將回應儲存至 debug_response.html 供檢查")
        return None

    courses = []

    # ==========================
    #  解析課程表（動態對應表頭）
    # ==========================
    for table in tables:
        rows = table.find_all('tr')
        if len(rows) <= 1:
            continue

        # 第一列當作表頭
        header_cells = rows[0].find_all(['th', 'td'])
        if not header_cells:
            continue

        headers = [cell.get_text(strip=True) for cell in header_cells]
        if all(not h for h in headers):
            continue

        # 找出「學分」欄位的所有可能表頭名稱
        credit_header_candidates = [h for h in headers if '學分' in h]

        # 解析每一筆資料列
        for row in rows[1:]:
            cells = row.find_all('td')
            if not cells:
                continue

            cell_texts = [c.get_text(strip=True) for c in cells]
            if not any(cell_texts):
                continue  # 全空列跳過

            # 依表頭對應每一欄
            row_data = {}
            for i, text in enumerate(cell_texts):
                if i < len(headers):
                    header_name = headers[i].strip()
                    if header_name:  # 忽略空白表頭
                        row_data[header_name] = text

            # 建立課程資料基本欄位
            course = {
                '學年學期': semester_label,
                '所屬年級': grade_name if grade_name else '全年級',
                '資料來源': url,
            }

            # 合併 row_data（讓原始表頭欄位保留）
            course.update(row_data)

            # 補救：若沒有「學分」這個精確欄位名稱，用含「學分」字樣的欄位當學分
            if '學分' not in course:
                for h in credit_header_candidates:
                    if h in course and course[h]:
                        course['學分'] = course[h]
                        break

            # 過濾掉「專任教師合計」「兼任教師合計」這類小計列
            course_name = course.get('課程名稱', '')
            if course_name in ['專任教師合計', '兼任教師合計']:
                continue

            # 再過濾一下：沒有課程名稱基本上就不是課程
            if not course_name:
                continue

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
        ('U4', '四年級'),
    ]

    # 定義學期
    semesters = [
        ('1', '第一學期'),
        ('2', '第二學期'),
    ]

    all_courses = []

    # 逐一爬取每個學期和年級的課程
    for semester_num, semester_name in semesters:
        print(f"\n{'=' * 50}")
        print(f"開始爬取 113學年度{semester_name}")
        print('=' * 50)

        for class_level, grade_name in grade_levels:
            print(f"\n正在爬取 {grade_name} 課程...")
            courses = get_course_history(
                school_year='113',
                semester=semester_num,
                department='06',
                class_level=class_level,
                grade_name=grade_name,
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

        # 按學期和年級組織課程資料
        output_data: dict = {}

        for course in all_courses:
            semester = course['學年學期']
            grade = course['所屬年級']

            if semester not in output_data:
                output_data[semester] = {}

            if grade not in output_data[semester]:
                output_data[semester][grade] = {
                    "課程數": 0,
                    "課程列表": [],
                }

            course_data = {
                "課程名稱": course.get('課程名稱', ''),
                "課號": course.get('課號', ''),
                "教師": course.get('教師', ''),
                "選別": course.get('選別', ''),
                "學分": course.get('學分', ''),
                "資料來源": course.get('資料來源', ''),
            }

            output_data[semester][grade]["課程列表"].append(course_data)
            output_data[semester][grade]["課程數"] += 1

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 50}")
        print(f"課程資料已儲存至 {output_file}")
        print(f"總計爬取 {len(all_courses)} 筆課程資料")
        print(f"{'=' * 50}")

        # 統計每個學期和年級的課程數量
        print("\n學期統計:")
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
        print("\n爬取失敗, 請檢查 debug_response.html 以了解問題")
