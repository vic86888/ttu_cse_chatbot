# web/app.py
import gradio as gr
import sys
import os

# 將專案根目錄加入 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import retrieve_docs
from llm.model import generate_answer

def query_llm(user_input, chat_history):
    """
    處理使用者輸入並返回 LLM 答案
    
    Args:
        user_input: 使用者的問題
        chat_history: 對話歷史記錄
        
    Returns:
        tuple: (空字串, 更新後的對話歷史)
    """
    if not user_input.strip():
        return "", chat_history
    
    # 1. 取得相關文件
    docs, sources_info = retrieve_docs(user_input)
    
    # 2. 生成回答
    answer = generate_answer(user_input, docs)
    
    # 3. 組合回答與來源資訊
    full_response = f"{answer}\n\n---\n\n### 📚 參考來源：\n{sources_info}"
    
    # 4. 更新對話歷史
    chat_history.append((user_input, full_response))
    
    return "", chat_history

def clear_chat():
    """清除對話歷史"""
    return None

# 自定義 CSS 樣式
custom_css = """
#chatbot {
    height: 600px;
    overflow-y: auto;
}
.message-wrap {
    max-width: 85% !important;
}
"""

# 建立 Gradio 介面
with gr.Blocks(css=custom_css, title="大同大學資工系問答機器人") as demo:
    gr.Markdown(
        """
        # 💬 大同大學資工系問答機器人
        
        歡迎使用本地 RAG + LLM 聊天機器人！請在下方輸入您的問題。
        
        **系統特色：**
        - 🔍 基於向量檢索的智能問答
        - 🧠 使用 Qwen3 本地模型
        - 📚 包含課程、新聞、教師資訊等資料
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="對話記錄",
                elem_id="chatbot",
                bubble_full_width=False,
                height=600,
                show_copy_button=True
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="請輸入您的問題",
                    placeholder="例如：資工系有哪些教授？",
                    scale=4,
                    lines=2
                )
                
            with gr.Row():
                submit_btn = gr.Button("🚀 送出", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ 清除對話", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### 💡 使用提示
                
                **可詢問的問題類型：**
                - 📰 最新消息與活動
                - 👨‍🏫 教師資訊
                - 📚 課程資訊
                - 📞 聯絡方式
                - 🏫 系所介紹
                
                **注意事項：**
                - 系統會根據資料庫內容回答
                - 首次載入模型需要一些時間
                - 支援繁體中文問答
                """
            )
            
            gr.Markdown(
                """
                ---
                ### ⚙️ 系統資訊
                
                - **LLM 模型**: Qwen3
                - **向量模型**: BAAI/bge-m3
                - **重排序模型**: BAAI/bge-reranker-base
                - **向量資料庫**: Chroma
                """
            )
    
    # 事件綁定
    submit_btn.click(
        fn=query_llm,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
        show_progress=True
    )
    
    user_input.submit(
        fn=query_llm,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
        show_progress=True
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=chatbot
    )
    
    gr.Markdown(
        """
        ---
        <center>
        Made with ❤️ by TTU CSE | Powered by Gradio & LangChain
        </center>
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允許外部訪問
        server_port=7860,        # 預設端口
        share=False,             # 不建立公開連結
        show_error=True,         # 顯示錯誤訊息
        show_api=False           # 不顯示 API 文件
    )
