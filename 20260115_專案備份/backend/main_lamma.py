from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
import re
from typing import AsyncGenerator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager

# RAG / LangChain 相關
from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langsmith import traceable

# ========= 全域設定 =========
# 注意：這裡保留原本後端用的相對路徑
DB_DIR = "../storage/chroma"
# DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# 全域物件（啟動時建立）
reranker: CrossEncoder | None = None
chains: dict | None = None  # {"rag": ..., "rewrite": ...}

# ========= Rerank / Retriever =========
def rerank_docs(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """用 cross-encoder 對候選文件重新排序，只保留前 top_n。"""
    global reranker
    if not docs:
        return []
    if reranker is None:
        raise RuntimeError("reranker 尚未初始化")

    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)  # numpy array

    scored = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    out: list[Document] = []
    for doc, s in scored[:top_n]:
        md = dict(doc.metadata) if doc.metadata else {}
        md["rerank_score"] = float(s)
        out.append(Document(page_content=doc.page_content, metadata=md))

    return out


def make_scored_retriever(vdb, k: int = 10):
    """Chroma + relevance score + CrossEncoder rerank"""
    # 先抓比較多，再給 reranker 挑前 k
    k_retrieve = max(k * 4, 100)

    def _retrieve(query: str):
        def as_docs(pairs):
            out = []
            for doc, score in pairs:
                md = dict(doc.metadata) if doc.metadata else {}
                md["relevance"] = float(score)
                out.append(Document(page_content=doc.page_content, metadata=md))
            return out

        pairs = vdb.similarity_search_with_relevance_scores(
            query,
            k=k_retrieve,
        )
        docs = as_docs(pairs)

        # 用 cross-encoder 重新排序，只保留前 k 個
        docs = rerank_docs(query, docs, top_n=k)
        return docs

    return RunnableLambda(_retrieve).with_config({
        "run_name": "ChromaRetriever+Reranker",
        "tags": ["retriever", "chroma", "with-scores", "rerank"],
        "metadata": {"k": k},
    })


# ========= Query Rewriter / RAG Chain =========
def build_chain():
    """建立兩條鏈：rewrite_chain（處理時間） + rag_chain（真正回答）"""

    # 1) LLM
    llm = ChatOllama(
        model="cwchang/llama-3-taiwan-8b-instruct:latest",
        temperature=0,
    ).with_config({
        "run_name": "Ollama-LLM",
        "tags": ["ollama", "qwen3", "local"],
        "metadata": {"provider": "ollama"},
    })

    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一個查詢改寫器。\n"
            "現在時間：{now}（時區：Asia/Taipei）。\n"
            "目前學期：{acad_term}\n\n"
            
            "【精準時間參考表】(請優先參考此表)\n"
            "1. **上週 / 上禮拜** 區間為：{last_week}\n"
            "2. **這週 / 本週** 區間為：{this_week}\n"
            "3. **下週 / 下禮拜** 區間為：{next_week}\n\n"
            
            "【參考知識】\n"
            "1. 民國紀年 = 西元年份 - 1911。\n"
            "2. 「113學年度」約為 2024/08 ~ 2025/07。\n\n"
            
            "【任務目標】\n"
            "將使用者問題中的「相對時間詞」替換為上述參考表中的具體日期區間。\n"
            "若使用者詢問的是具體日期，請直接回答具體日期。\n"
            "請包含週末(週一至週日)。\n\n"
            "若無法直接抄寫，請根據現有的資訊進行計算出具體日期"
            "**針對「學期」相關詞彙，請保持原樣，不要改寫。**\n\n"
            
            "【改寫範例】\n"
            "- 輸入：「下週有什麼演講？」(假設參考表顯示下週為 114年1月6日至12日)\n"
            "  輸出：「114年1月6日至1月12日有什麼演講？」\n\n"
            
            "- 輸入：「上禮拜的紀錄」\n"
            "  輸出：「{last_week}的紀錄」(直接填入上週區間)\n\n"

            "- 輸入：「113上學期有什麼課？」 -> 輸出：「113上學期有什麼課？」(保持原樣)\n"
            "- 輸入：「2024年行事曆」 -> 輸出：「2024年行事曆」(使用者輸入數字，保持原樣)\n"
            "- 輸入：「今年寒假什麼時候？」 -> 輸出：「114年寒假什麼時候？」(相對時間 -> 民國年)\n"
            "- 輸入：「明年畢業典禮？」 -> 輸出：「115年畢業典禮？」(相對時間 -> 民國年)\n"
            "- 輸入：「日本的姊妹校有哪些？」 -> 輸出：「日本的姊妹校有哪些？」(無時間詞，原樣輸出)\n"
            "- 輸入：「大一上學期有哪些必修？」 -> 輸出：「大一上學期有哪些必修？」(保留原詞，不需補年份)\n" 
            "- 輸入：「下學期選課時間？」 -> 輸出：「下學期選課時間？」(保留原詞)\n\n"

            "【執行規則】\n"
            "1. 如果使用者的提問參考資料有提供請**直接抄寫**【精準時間參考表】中的日期區間來替換對應詞彙，不要自己計算。\n"
            "2. 若無法直接抄寫，請根據現有的資訊進行計算出具體日期\n"
            "3. 若使用者詢問的是具體日期，請直接回答具體日期。\n"
            "4. 若問題中已有明確年份數字，原封不動保留。\n"
            "5. 僅輸出改寫後的句子。\n"
        ),
        ("human", "{query}")
    ]).with_config({
        "run_name": "TemporalQueryRewriter",
        "tags": ["query-rewrite", "temporal"],
    })

    rewrite_chain = (rewrite_prompt | llm | StrOutputParser()).with_config({
        "run_name": "RewriteChain",
        "tags": ["chain", "rewrite"],
    })

    # 1.8) ✨ 定義文件格式 (讓 LLM 看到網址)
    # 🟢 修改 3: 新增 document_prompt
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="【來源網址】: {source}\n【內容】:\n{page_content}\n\n"
    )

    # 2) RAG 回答提示詞
    # 🟢 修改 4: 更新 Prompt，要求 Markdown 連結與嚴格忠於文件
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "現在時間：{now}\n\n"
        "目前學期：{acad_term}\n\n"
        
        "【參考知識：校務常識與定義】\n"
        "1. **年份換算公式**：民國年份 = 西元年份 - 1911。（例：西元2025年 = 民國114年）\n"
        "2. **學年度定義 (關鍵)**：\n"
        "   - 一個學年度橫跨兩個民國年份。\n"
        "   - **113學年度** = 民國113年8月1日 ~ 民國114年7月31日。\n"
        "   - **114學年度** = 民國114年8月1日 ~ 民國115年7月31日。\n"
        "   - **判斷範例**：若日期為「民國115年1月」，它仍屬於「114學年度」（的第一學期末），**絕非**115學年度。\n"
        "3. **學期月份定義**：\n"
        "   - **第一學期 (上學期)**：通常包含 8月、9月、10月、11月、12月，以及**隔年的 1月**。\n"
        "     (例：**114學年度第一學期** = 民國114年8月 至 民國115年1月)\n"
        "   - **第二學期 (下學期)**：通常包含隔年的 2月、3月、4月、5月、6月、7月。\n"
        "     (例：**114學年度第二學期** = 民國115年2月 至 民國115年7月)\n"
        # 🔥 新增這點：教它區分「開學日」與「學期開始」 🔥
        "4. **開學日定義**：\n"
        "   - **學期開始日**：通常為 8月1日 (上學期) 或 2月1日 (下學期)，這只是行政上的學期起點。\n"
        "   - **開學日 (上課開始)**：通常為 9月中旬 (上學期) 或 2月中旬 (下學期)，這才是學生真正開始上課的日子。\n"
        "   - **回答原則**：若使用者詢問「開學日」，請優先回答「上課開始」的日期，而非「學期開始」的日期。\n"
        "5. **實例判斷**：若行事曆顯示「12月29日 至 1月11日」，且位於 114 學年度，則此日期區間屬於「114 學年度第一學期」。\n"
        # 🔥 新增這行：解決 Q2 系辦位置亂列一通的問題
        "6. **地點常識**：「資工系辦公室」通常指系行政中心（系務助理辦公處），位於**電機大樓 406 室**，而非個別教授的辦公室。\n\n"
        "7. **身分定義**：「資工系學生」即為「大同大學學士班學生」，適用所有全校性學則。\n\n"
        # 🔥 新增這點：教它「補假」的邏輯 🔥
        "8. **補假/換休日定義**：學校活動（如校慶、運動會）若遇假日，其「補假」或「換休日」可能會安排在**數週甚至數月之後**（例如：11月的活動，隔年1月才補假）。若文件中明確記載「某日為某活動之補假」，請直接採信該日期，無需因為日期差距而懷疑。\n"
        "9. **規章引用原則**：當使用者詢問「資工系規定」，若文件中僅有「全校性規定」，請**直接引用**全校規定回答，並說明「依據全校性規定...」，**嚴禁**以「未提及資工系」為由拒答。\n\n"
        
        "你是大同大學資工系問答機器人。\n"
        "你的任務是根據下方的【參考文件】來回答使用者的問題。\n\n"
        
        "【嚴格遵守規則】\n"
        "1. **絕對忠於文件**：嚴格禁止使用你的「外部常識」或「預訓練知識」來回答（例如：不要用一般的大學常識來回答本系的規定）。如果文件沒寫，就誠實說「文件中未提及相關資訊」。\n"
        "2. **年份格式**：回答時的年份格式（民國/西元），請優先跟隨【參考文件】中的原始寫法。不需要刻意解釋換算過程。\n"
        "3. **引用來源格式 (重要)**：\n"
        "   - 文件中已提供【來源網址】，請務必使用 Markdown 超連結格式來標註來源。\n"

        "   - 連結文字請優先使用該文件的「規章名稱」或「標題」。\n"
        "   - 格式範例：`[點擊查看來源](https://www.ttu.edu.tw/rule.pdf)`。\n"
        "   - 若無法確定標題，則使用 `[原始文件](網址)`。\n"
        
        "   - 請在回答的相關段落後，或是在回答結尾的「參考來源」清單中附上連結。\n"
        "   - 使用 Markdown 格式（## 標題, - 列表, **粗體**）：\n"
        "- 使用 ## 二級標題來分隔不同主題\n"
        "- 使用列表 (- 或 1.) 來呈現多個項目\n"
        "- 使用 **粗體** 來強調重要資訊\n"
        "- 使用程式碼區塊 ```語言 來展示程式碼\n"
        "- 使用表格來呈現結構化資料\n\n"
        "4. **網址絕對精確 (Critical)**：\n"
        "   - **嚴格禁止縮減**：即使網址非常長，也必須**完整輸出**，絕對不要使用 `...` 或 `（完整連結略）` 等方式省略。\n"
        "   - **嚴格禁止**修改、拼湊或創造網址。\n"
        "   - 網址必須**逐字複製**自【來源網址】欄位，連一個字元都不能改。\n"
        "   - 錯誤範例：把 `cse.ttu.edu.tw` 的路徑接到 `tchinfo.ttu.edu.tw` 上。\n\n"
        
        "【參考文件】\n"
        "{context}"),
        ("human", "{input}")
    ]).with_config({
        "tags": ["chain", "stuff"],
    })

    # 2) RAG 回答提示詞
    # 🟢 修改 4: 更新 Prompt，要求 Markdown 連結與嚴格忠於文件
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "現在時間：{now}\n\n"
        "目前學期：{acad_term}\n\n"
        
        "【參考知識：校務常識與定義】\n"
        "1. **年份換算公式**：民國年份 = 西元年份 - 1911。（例：西元2025年 = 民國114年）\n"
        "2. **學年度定義 (關鍵)**：\n"
        "   - 一個學年度橫跨兩個民國年份。\n"
        "   - **113學年度** = 民國113年8月1日 ~ 民國114年7月31日。\n"
        "   - **114學年度** = 民國114年8月1日 ~ 民國115年7月31日。\n"
        "   - **判斷範例**：若日期為「民國115年1月」，它仍屬於「114學年度」（的第一學期末），**絕非**115學年度。\n"
        "3. **學期月份定義**：\n"
        "   - **第一學期 (上學期)**：通常包含 8月、9月、10月、11月、12月，以及**隔年的 1月**。\n"
        "   - **第二學期 (下學期)**：通常包含隔年的 2月、3月、4月、5月、6月、7月。\n"
        # 🔥 新增這點：教它區分「開學日」與「學期開始」 🔥
        "4. **開學日定義**：\n"
        "   - **學期開始日**：通常為 8月1日 (上學期) 或 2月1日 (下學期)，這只是行政上的學期起點。\n"
        "   - **開學日 (上課開始)**：通常為 9月中旬 (上學期) 或 2月中旬 (下學期)，這才是學生真正開始上課的日子。\n"
        "   - **回答原則**：若使用者詢問「開學日」，請優先回答「上課開始」的日期，而非「學期開始」的日期。\n"
        "5. **實例判斷**：若行事曆顯示「12月29日 至 1月11日」，且位於 114 學年度，則此日期區間屬於「114 學年度第一學期」。\n"
        # 🔥 新增這行：解決 Q2 系辦位置亂列一通的問題
        "6. **地點常識**：「資工系辦公室」通常指系行政中心（系務助理辦公處），位於**電機大樓 406 室**，而非個別教授的辦公室。\n\n"
        "7. **身分定義**：「資工系學生」即為「大同大學學士班學生」，適用所有全校性學則。\n\n"
        # 🔥 新增這點：教它「補假」的邏輯 🔥
        "8. **補假/換休日定義**：學校活動（如校慶、運動會）若遇假日，其「補假」或「換休日」可能會安排在**數週甚至數月之後**（例如：11月的活動，隔年1月才補假）。若文件中明確記載「某日為某活動之補假」，請直接採信該日期，無需因為日期差距而懷疑。\n"
        "9. **規章引用原則**：當使用者詢問「資工系規定」，若文件中僅有「全校性規定」，請**直接引用**全校規定回答，並說明「依據全校性規定...」，**嚴禁**以「未提及資工系」為由拒答。\n\n"
        
        "你是大同大學資工系問答機器人。\n"
        "你的任務是根據下方的【參考文件】來回答使用者的問題。\n\n"
        
        "【回答規則】\n"
        "1. **年份格式 (關鍵)**：涉及年份或學年度時，請務必優先使用「民國紀年」（例如：113學年、114年），嚴禁自動轉換為西元（如 2025年、2026年），除非文件中僅提供西元格式。這是為了符合校內規章慣例。\n"
        "2. **完整語句**：請以「主詞 + 謂語」的完整句子回答，語氣需肯定且專業。例如：「畢業門檻為128學分」，而非僅回答「128」。\n"
        "3. **綜合推論**：若答案散落在【參考文件】的不同段落，請嘗試將其整合。若資訊不完整但可合理推斷，請回答並加上「根據文件推測...」；不要因為沒有「逐字對應」就拒絕回答。\n"
        "4. **嚴禁瞎編**：若【參考文件】中完全沒有提及問題所需的關鍵資訊（特別是具體日期、人名、電話），請直接回答「文件中沒有相關資訊」，絕對不可自行生成數據。\n"
        "5. **多項條列**：若答案包含多個項目，請列出所有項目並用頓號或逗號隔開，保持語句通順。\n"
        "6. **禁止輸出思考**：請直接輸出最終答案，不要包含 `<think>` 標籤或推理過程。\n"
        
        "【參考文件】\n"
        "{context}"),
        ("human", "{input}")
    ]).with_config({
        "tags": ["chain", "stuff"],
    })

    # 3) combine documents chain
    doc_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        # prompt=eval_prompt,
        document_prompt=document_prompt,   # 🟢 修改 5: 注入 document_prompt
        document_variable_name="context"
    ).with_config({
        "run_name": "StuffDocumentsChain",
        "tags": ["chain", "stuff"],
    })

    # 4) 向量庫 & 檢索器（含分數）
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},  # 若沒有 GPU 可改成 "cpu"
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    scored_retriever = make_scored_retriever(vectordb, k=10)
    # create_retrieval_chain 期望 retriever 接收整個 input dict
    retriever_runnable = itemgetter("input") | scored_retriever

    # 5) RAG 鏈（retriever + combine_docs）
    rag_chain = create_retrieval_chain(retriever_runnable, doc_chain).with_config({
        "run_name": "CampusRAG",
        "tags": ["campus-rag", "api"],
    })

    return {
        "rag": rag_chain,
        "rewrite": rewrite_chain,
    }


# ========= 工具函式 =========
def extract_clean_query(text: str) -> str:
    """從 rewriter 輸出中抽出『真正要拿去當 query 的那句話』"""
    if not text:
        return ""

    # 1) 若有 <answer>...</answer>，優先用裡面的內容
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2) 把 <think>...</think> 整塊砍掉
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 3) 只留最後一行（通常是實際問題）
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1]


def remove_thinking_tags(text: str) -> str:
    """從回應文本中移除模型的思考過程（<think>...</think> 標籤）"""
    if not text:
        return ""
    # 移除所有 <think>...</think> 標籤及其內容
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


@traceable(name="API-Ask", run_type="chain", metadata={"app": "campus_rag_api"})
def ask(chain_dict: dict, q: str):
    """執行查詢：先用 LLM 做時間改寫，再丟給 RAG"""
    rag_chain = chain_dict["rag"]
    rewrite_chain = chain_dict["rewrite"]

    now = datetime.now(ZoneInfo("Asia/Taipei"))
    roc_year = now.year - 1911

    # --- 🔥 新增：Python 精準計算週區間 (Monday-Sunday) ---
    # isoweekday(): 1=週一, 7=週日
    weekday = now.isoweekday() 
    
    # 1. 本週 (This Week)
    # 本週一 = 今天 - (今天星期幾 - 1)
    start_this = now - timedelta(days=weekday - 1)
    end_this = start_this + timedelta(days=6)
    
    # 2. 上週 (Last Week)
    start_last = start_this - timedelta(days=7)
    end_last = end_this - timedelta(days=7)
    
    # 3. 下週 (Next Week)
    start_next = start_this + timedelta(days=7)
    end_next = end_this + timedelta(days=7)

    # 格式化為字串 (優先轉為民國年，方便 LLM 理解)
    def fmt_range(s, e):
        sy, sm, sd = s.year - 1911, s.month, s.day
        ey, em, ed = e.year - 1911, e.month, e.day
        return f"{sy}年{sm}月{sd}日 至 {ey}年{em}月{ed}日"

    this_week_str = fmt_range(start_this, end_this)
    last_week_str = fmt_range(start_last, end_last)
    next_week_str = fmt_range(start_next, end_next)

    m, d = now.month, now.day

    # 學年學期計算：8/1 新學年；2/1 第二學期
    if (m, d) >= (8, 1):
        acad_year = roc_year
        sem = "第一學期"
    elif (m, d) >= (2, 1):
        acad_year = roc_year - 1
        sem = "第二學期"
    else:
        acad_year = roc_year - 1
        sem = "第一學期"

    acad_term = f"{acad_year}學年{sem}"
    now_str = f"民國{roc_year}年{m}月{d}日 {now.strftime('%H:%M')}"

    # # 先讓 rewrite_chain 把相對時間改寫
    # try:
    #     raw = rewrite_chain.invoke({
    #         "query": q,
    #         "now": now.strftime("%Y年%m月%d日 %H:%M"),
    #         "acad_term": acad_term,
    #         "this_week": this_week_str,
    #         "last_week": last_week_str,
    #         "next_week": next_week_str,
    #     })
    #     raw = raw.strip()
    #     print(f"[DEBUG] rewrite raw: {raw!r}")  # 建議先看模型原始輸出
    #     rewritten_q = extract_clean_query(raw)
    # except Exception as e:
    #     print(f"[DEBUG] rewrite failed: {e!r}")
    #     rewritten_q = q

    # if not rewritten_q:
    #     rewritten_q = q

    # print(f"[DEBUG] rewritten query: {rewritten_q!r}")

    # ✅ 測試用暫時方案：
    print("⚠️ [TEST MODE] 強制跳過 Rewrite，直接使用原始問題")
    rewritten_q = q 
    
    # ==========================================

    print(f"[DEBUG] rewritten query: {rewritten_q!r}")


    # 再把改寫後的 query 丟進 RAG
    result = rag_chain.invoke({
        "input": rewritten_q,
        "now": now_str,
        "acad_term": acad_term,
        "original_query": q,
        "rewritten_query": rewritten_q,
    })

    # ================= 🔍 DEBUG 專區 =================
    # 1. 檢查到底抓到了幾份文件？內容是什麼？
    retrieved_docs = result.get("context", [])
    print(f"\n[DEBUG] 檢索到的文件數量: {len(retrieved_docs)}")
    if len(retrieved_docs) > 0:
        print(f"[DEBUG] 第一份文件標題/來源: {retrieved_docs[0].metadata.get('source', '未知')}")
        print(f"[DEBUG] 第一份文件內容前 100 字: {retrieved_docs[0].page_content[:100]}...")
    
    # 2. 檢查模型最原始的回答到底是什麼？
    raw_answer = result.get("answer", "")
    print(f"\n[DEBUG] 🤖 模型原始回答 (Raw Answer) >>>")
    print(f"'{raw_answer}'")  # 用引號包起來，看是不是空字串
    print(f"<<< (長度: {len(raw_answer)})")
    # ================================================

    # ⭐⭐ 在這裡把 <think>...</think> 移掉，只留下真正要顯示的回答
    raw_answer = result.get("answer", "")
    thinking_process = ""

    # 1. 嘗試提取 <think> 區塊並印出
    think_match = re.search(r"<think>(.*?)</think>", raw_answer, re.DOTALL)
    if think_match:
        thinking_process = think_match.group(1).strip()
        print("\n" + "="*40)
        print("🧠 模型思考過程 (Thinking Process):")
        print(thinking_process)
        print("="*40 + "\n")
    else:
        print("\n[INFO] 本次回答未包含 <think> 標籤\n")

    # 2. 移除標籤，只保留乾淨回答給前端
    clean_answer = remove_thinking_tags(raw_answer)
    result["answer"] = clean_answer
    result["thinking_process"] = thinking_process  # 🔥 新增這個欄位！

    return result


# ========= FastAPI App & Lifespan =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用啟動 / 關閉生命週期"""
    global reranker, chains

    print("🔄 載入 reranker 模型...")
    reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")
    print("🔄 建立 RAG chains...")
    chains = build_chain()
    print("✅ 模型載入完成！")

    yield

    print("👋 關閉應用...")


app = FastAPI(title="TTU CSE Chatbot API", lifespan=lifespan)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= 基本路由 =========
@app.get("/")
async def root():
    return {"message": "TTU CSE Chatbot API is running"}


@app.get("/favicon.ico")
async def favicon():
    """防止 favicon 404 錯誤"""
    return {"message": "No favicon"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_ready": chains is not None,
    }


# ========= SSE 串流 =========
async def generate_stream(message: str) -> AsyncGenerator[str, None]:
    """生成 SSE 串流回應（使用新的 RAG + rewriter 邏輯）"""
    try:
        global chains
        if chains is None:
            raise RuntimeError("RAG chains 尚未初始化")

        # 在背景 thread 跑完整 ask（含 rewrite + RAG）
        result = await asyncio.to_thread(ask, chains, message)
        response_text = result.get("answer", "")

        # 逐字串流
        for char in response_text:
            payload = {"content": char, "done": False}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)

        # 發送完成訊號
        yield f"data: {json.dumps({'content': '', 'done': True}, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_msg = f"抱歉，處理您的問題時發生錯誤：{str(e)}"
        payload = {"content": error_msg, "done": True, "error": True}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/chat/stream")
async def chat_stream(message: str):
    """SSE 串流聊天端點（GET /api/chat/stream?message=...）"""
    return StreamingResponse(
        generate_stream(message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ========= 非串流 API =========
@app.post("/api/chat")
async def chat(request: dict):
    """非串流聊天端點（使用新的 RAG + rewriter 邏輯）"""
    try:
        global chains
        if chains is None:
            raise RuntimeError("RAG chains 尚未初始化")

        message = request.get("message", "")

        result = await asyncio.to_thread(ask, chains, message)

        return {
            "response": result.get("answer", ""),
            "original_query": result.get("original_query", message),
            "rewritten_query": result.get("rewritten_query", message),
            "now": result.get("now"),
            "acad_term": result.get("acad_term"),
            "sources": [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance": doc.metadata.get("relevance"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                }
                for doc in result.get("context", [])
            ],
        }
    except Exception as e:
        return {
            "response": f"抱歉，處理您的問題時發生錯誤：{str(e)}",
            "error": True,
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_lamma:app", host="0.0.0.0", port=8000, reload=False)
