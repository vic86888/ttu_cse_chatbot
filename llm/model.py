# llm/model.py
"""
LLM 模型模組
負責生成回答
"""
from typing import List
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain

# 全域變數
llm_chain = None


def initialize_llm():
    """初始化 LLM 和提示詞鏈"""
    global llm_chain
    
    if llm_chain is None:
        # 1) LLM
        llm = ChatOllama(
            model="qwen3:latest",
            temperature=0,
        )
        
        # 2) 提示詞
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是大同大學資工系問答機器人。\n"
             "學年等於民國紀年,114學年就是2025年。"
             "你會得到跟問題相關的文件,你只依據提供的文件內容回答問題,"
             "若無法從文件中找到答案,請清楚說明。請以繁體中文作答。\n\n"
             "{context}"),
            ("human", "{input}")
        ])
        
        # 3) 建立文件處理鏈
        llm_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)


def generate_answer(query: str, docs: List[Document]) -> str:
    """
    根據查詢和文件生成答案
    
    Args:
        query: 使用者查詢
        docs: 檢索到的文件列表
        
    Returns:
        生成的答案字串
    """
    # 確保已初始化
    initialize_llm()
    
    # 取得台北時間並加入查詢中
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    roc_year = now.year - 1911
    today_roc = f"{roc_year}年{now.month}月{now.day}日"
    
    # 將時間資訊附加到問題前面
    timestamped_query = f"[當前時間: {today_roc}] {query}"
    
    try:
        # 使用 LLM 鏈生成答案
        answer = llm_chain.invoke({
            "input": timestamped_query,
            "context": docs
        })
        
        return answer
    except Exception as e:
        return f"⚠️ 生成答案時發生錯誤：{str(e)}\n\n請稍後再試或重新啟動系統。"
