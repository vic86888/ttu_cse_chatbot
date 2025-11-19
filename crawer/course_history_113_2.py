from course_history_113_1 import get_course_history
import json
import os

print("=== 測試學期參數 ===\n")

# 測試第二學期
print("開始爬取課程資料 - 113學年度第二學期")
print("-" * 50)

# 定義要爬取的年級
grade_levels = [
    ('U1', '一年級'),
    ('U2', '二年級'),
    ('U3', '三年級'),
    ('U4', '四年級')
]

all_courses = []

# 逐一爬取每個年級的課程
for class_level, grade_name in grade_levels:
    print(f"\n正在爬取 {grade_name} 課程...")
    courses = get_course_history(
        school_year='113', 
        semester='2', 
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
    output_file = os.path.join(data_dir, 'course_history_113_2.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_courses, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"課程資料已儲存至 {output_file}")
    print(f"總計爬取 {len(all_courses)} 筆課程資料")
    print(f"{'='*50}")
    
    # 統計每個年級的課程數量
    print("\n年級統計:")
    from collections import Counter
    grade_count = Counter(course['所屬年級'] for course in all_courses)
    for grade, count in sorted(grade_count.items()):
        print(f"  {grade}: {count} 筆")
    
    if len(all_courses) > 0:
        print("\n前三筆課程資料範例:")
        for i, course in enumerate(all_courses[:3], 1):
            print(f"\n第 {i} 筆:")
            print(json.dumps(course, ensure_ascii=False, indent=2))
