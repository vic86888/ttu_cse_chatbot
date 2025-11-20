# 大同大學資工系問答機器人 - Gradio 網頁介面

這是基於 Gradio 框架建立的前端介面,用於與本地 RAG + LLM 系統互動。

## 📁 專案結構

```
ttu_cse_chatbot/
│
├── web/
│   ├── app.py              # Gradio 主程式
│   └── gradio.md           # 架構規範文件
│
├── rag/                    # RAG 相關模組
│   ├── __init__.py
│   └── retriever.py        # 向量檢索器
│
├── llm/                    # LLM 相關模組
│   ├── __init__.py
│   └── model.py            # 模型推理
│
├── data/                   # 資料檔案
├── storage/chroma/         # 向量資料庫
└── requirements.txt        # Python 套件依賴
```

## 🚀 快速開始

### 1. 安裝依賴套件

```powershell
pip install gradio>=4.0.0
```

或安裝所有依賴:

```powershell
pip install -r requirements.txt
```

### 2. 確認環境

確保以下項目已準備好:

- ✅ Ollama 已安裝並運行 `qwen3:latest` 模型
- ✅ 向量資料庫已建立在 `storage/chroma/` 目錄
- ✅ GPU 驅動已安裝 (建議使用 CUDA)

### 3. 啟動網頁介面

在專案根目錄執行:

```powershell
python web/app.py
```

或從 web 目錄執行:

```powershell
cd web
python app.py
```

### 4. 訪問介面

啟動後,在瀏覽器中開啟:

```
http://localhost:7860
```

## 💡 功能特色

### 🎯 核心功能

- **智能問答**: 基於 RAG 技術的精準回答
- **來源追蹤**: 顯示答案的參考文件與分數
- **對話記錄**: 保存完整的問答歷史
- **即時處理**: 實時向量檢索與模型推理

### 📊 系統架構

1. **前端層** (Gradio)
   - 使用者介面
   - 對話管理
   - Markdown 渲染

2. **檢索層** (RAG)
   - 向量相似度搜尋
   - Cross-encoder 重排序
   - 智能文件過濾

3. **生成層** (LLM)
   - Qwen3 本地模型
   - 上下文整合
   - 答案生成

## 🔧 配置說明

### 模型設定

在 `llm/model.py` 中可修改:

```python
llm = ChatOllama(
    model="qwen3:latest",  # 可改為其他 Ollama 模型
    temperature=0,          # 調整創造性 (0-1)
)
```

### 檢索參數

在 `rag/retriever.py` 中可調整:

```python
def retrieve_docs(query: str, k: int = 10):  # k 為返回文件數
    k_retrieve = max(k * 4, 20)  # 初始檢索數量
```

### 網頁設定

在 `web/app.py` 的 `demo.launch()` 中可修改:

```python
demo.launch(
    server_name="0.0.0.0",  # 允許外部訪問
    server_port=7860,        # 更改端口
    share=False,             # True 可建立公開連結
)
```

## 📝 使用範例

### 問題範例

1. **最新消息查詢**
   - "最近有什麼活動？"
   - "最新的系上新聞是什麼？"

2. **課程資訊**
   - "資料結構的課程內容有哪些？"
   - "114學年有哪些課程？"

3. **教師資訊**
   - "資工系有哪些教授？"
   - "王教授的研究領域是什麼？"

4. **聯絡資訊**
   - "系辦的電話是多少？"
   - "如何聯絡資工系？"

## 🛠️ 故障排除

### 常見問題

**Q1: 無法啟動 Gradio**
```
解決方法: 確認已安裝 gradio 套件
pip install gradio>=4.0.0
```

**Q2: 找不到模型**
```
解決方法: 確認 Ollama 已安裝並下載模型
ollama pull qwen3:latest
```

**Q3: CUDA 記憶體不足**
```
解決方法: 在 retriever.py 和 model.py 中將 device="cuda" 改為 device="cpu"
```

**Q4: 向量資料庫錯誤**
```
解決方法: 確認 storage/chroma/ 目錄存在且有資料
可執行 ingest.py 重新建立資料庫
```

## 📚 技術堆疊

- **前端框架**: Gradio 4.0+
- **LLM**: Ollama (Qwen3)
- **向量模型**: BAAI/bge-m3
- **重排序**: BAAI/bge-reranker-base
- **向量資料庫**: Chroma
- **框架**: LangChain

## 🔒 安全性

- 僅本地部署,不暴露外網
- 不收集使用者資料
- 所有處理在本地完成

## 📞 支援

如有問題,請聯繫開發團隊或查看:
- `query.py` - 原始命令列版本
- `gradio.md` - 架構規範文件

---

**Made with ❤️ by TTU CSE**
