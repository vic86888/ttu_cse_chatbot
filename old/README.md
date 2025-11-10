# TTU CSE Chatbot

一個簡易的 RAG (Retrieval-Augmented Generation) 專案，透過向量資料庫實作 PDF 文件的問答系統。  
使用者可以上傳 PDF，系統會自動切塊、轉換為向量並存入資料庫，之後即可進行查詢與問答。  

---

## 📂 專案結構

- **data/**  
  測試用 PDF 檔案存放資料夾  

- **ingest.py**  
  負責檔案抽取、切塊、轉換向量，並儲存至向量資料庫  

- **inspect_chroma.py**  
  檢視向量資料庫中指定 PDF 的切塊內容  

- **query.py**  
  問答模組，使用者可輸入問題，系統會根據向量檢索回覆答案  

- **requirements.txt**  
  專案所需套件與版本管理  

---
