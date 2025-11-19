import json
import os

def organize_courses_by_type_and_semester():
    """
    讀取課程資料並按照選別(必修/選修)、學年學期和年級分類課程名稱
    """
    # 定義檔案路徑(專案根目錄的 data 資料夾)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    file1 = os.path.join(data_dir, "course_history_113_1.json")
    file2 = os.path.join(data_dir, "course_history_113_2.json")
    
    # 儲存結果的列表
    organized_data = []
    
    # 用於分類的字典
    course_dict = {}
    
    # 讀取兩個檔案
    files_to_read = [file1, file2]
    
    for filename in files_to_read:
        if not os.path.exists(filename):
            print(f"警告: 找不到檔案 {filename}")
            continue
        
        print(f"正在讀取 {filename}...")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                courses = json.load(f)
            
            # 處理每一筆課程資料
            for course in courses:
                semester = course.get('學年學期', '')
                course_type = course.get('選別', '')
                course_name = course.get('課程名稱', '')
                grade = course.get('所屬年級', '')
                
                # 只處理有選別和學年學期的資料
                if semester and course_type and course_name:
                    # 建立唯一鍵值 (加入年級分類)
                    key = (course_type, semester, grade)
                    
                    # 初始化字典
                    if key not in course_dict:
                        course_dict[key] = set()  # 使用 set 避免重複
                    
                    # 加入課程名稱
                    course_dict[key].add(course_name)
            
            print(f"成功讀取 {len(courses)} 筆課程資料")
            
        except Exception as e:
            print(f"讀取 {filename} 時發生錯誤: {e}")
            continue
    
    # 將字典轉換為列表格式
    for (course_type, semester, grade), course_names in sorted(course_dict.items()):
        organized_data.append({
            "選別": course_type,
            "學年學期": semester,
            "所屬年級": grade,
            "課程名稱": sorted(list(course_names))  # 轉換為排序後的列表
        })
    
    # 儲存為 JSON 檔案(使用已定義的 data_dir)
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, "organized_courses.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(organized_data, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 50)
        print(f"成功建立 {len(organized_data)} 個分類")
        print(f"資料已儲存至 {output_file}")
        print("=" * 50)
        
        # 顯示統計資訊
        print("\n分類統計:")
        for item in organized_data:
            course_type = item['選別']
            semester = item['學年學期']
            grade = item['所屬年級']
            count = len(item['課程名稱'])
            print(f"  {semester} - {grade} - {course_type}: {count} 門課程")
        
        # 顯示範例資料
        if organized_data:
            print("\n第一筆資料範例:")
            print(json.dumps(organized_data[0], ensure_ascii=False, indent=2))
        
        return organized_data
        
    except Exception as e:
        print(f"寫入 JSON 檔案失敗: {e}")
        return None

if __name__ == "__main__":
    print("=== 課程資料整理程式 ===\n")
    organize_courses_by_type_and_semester()
