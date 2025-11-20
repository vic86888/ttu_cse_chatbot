
## **Gradio 架構使用規範（本地 RAG + LLM）**

### **1️⃣ 專案目錄建議**

```
project/
│
├─ app.py              # Gradio 主程式
├─ rag/                # RAG 相關程式碼
│   ├─ retriever.py
│   └─ embedder.py
├─ llm/                # LLM 相關程式碼
│   └─ model.py
├─ data/               # 本地資料
│   └─ documents.json
└─ requirements.txt
```

### **2️⃣ Python 套件**

* `gradio`：前端介面
* `transformers` / `llama_index` / `sentence-transformers`：LLM 與向量檢索
* `faiss` 或 `chroma`：本地向量庫
* 其他依需求：`torch`, `numpy`, `markdown`

```txt
gradio>=3.40
torch
transformers
sentence-transformers
faiss-cpu
```

---

### **3️⃣ Gradio 使用規範**

#### **基本流程**

1. 使用者在 Gradio Textbox 輸入 query
2. 呼叫 RAG retriever 取得相關文件
3. 將 query + documents 組成 prompt 給 LLM
4. LLM 輸出回答（Markdown 格式）
5. Gradio Markdown 或 HTML 渲染輸出

#### **介面組件建議**

* `Textbox`：用戶輸入
* `Button`：觸發查詢
* `Markdown`：顯示 LLM 回答（可呈現格式化文本）
* 可選：`Chatbot` 組件呈現對話式介面

---

### **4️⃣ Gradio 主程式示意**

```python
import gradio as gr
from rag.retriever import retrieve_docs
from llm.model import generate_answer

def query_llm(user_input):
    # 1. 取得相關文件
    docs = retrieve_docs(user_input)
    
    # 2. 生成回答
    answer = generate_answer(user_input, docs)
    
    return answer

with gr.Blocks() as demo:
    gr.Markdown("# 本地 RAG + LLM 聊天機器人")
    
    with gr.Row():
        user_input = gr.Textbox(label="請輸入你的問題")
        submit_btn = gr.Button("送出")
    
    output_md = gr.Markdown(label="回答")
    
    submit_btn.click(fn=query_llm, inputs=user_input, outputs=output_md)

demo.launch()
```

---

### **5️⃣ 使用規範要點**

1. **Markdown 支援**：Gradio 的 Markdown 組件可直接呈現 LLM 輸出的格式化文本。
2. **本地部署**：不依賴雲端，RAG 和 LLM 模型都在本地運行。
3. **非同步建議**：若模型較大，可用 `threading` 或 `async` 避免 UI 卡住。
4. **版本控制**：`requirements.txt` 固定版本，方便團隊快速部署。
5. **安全性**：僅本地使用，不暴露 HTTP API 給外網。
