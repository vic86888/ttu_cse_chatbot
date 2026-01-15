# query.py
from typing import Set
import os
import re

from datetime import datetime
from zoneinfo import ZoneInfo  # 新增這行

from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langsmith import traceable
from sentence_transformers import CrossEncoder
from langchain_core.output_parsers import StrOutputParser

# Rich 套件用於美化終端輸出
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

Prompt.prompt_suffix = ""  # 或 " " 之類的，避免預設的冒號

console = Console()

DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"

# EVENT_KEYWORDS = [
#     "新聞", "消息", "news", "最新",
#     "最近", "活動", "說明會", "講座", "論壇", "營隊", "徵才"
# ]

from langchain_core.documents import Document

def rerank_docs(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """用 cross-encoder 對候選文件重新排序，只保留前 top_n。"""
    if not docs:
        return []

    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)  # 長度 = len(docs) 的 numpy array

    scored = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    out: list[Document] = []
    for doc, s in scored[:top_n]:
        md = dict(doc.metadata) if doc.metadata else {}
        md["rerank_score"] = float(s)  # 之後 debug 好看一點
        out.append(Document(page_content=doc.page_content, metadata=md))

    return out

def make_scored_retriever(vdb, k: int = 10):
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

        # q = (query or "").lower()
        # prefer_news = any(kw in q for kw in EVENT_KEYWORDS)

        docs: list[Document] = []
        # if prefer_news:
        #     news_pairs = vdb.similarity_search_with_relevance_scores(
        #         query, k=k_retrieve, filter={"content_type": "news"}
        #     )
        #     docs.extend(as_docs(news_pairs))

        #     if len(docs) < k_retrieve:
        #         more_pairs = vdb.similarity_search_with_relevance_scores(
        #             query, k=k_retrieve
        #         )
        #         docs.extend(as_docs(more_pairs))

        #     # 去重（source+article_id 或 source+idx）
        #     seen = set()
        #     uniq = []
        #     for d in docs:
        #         md = d.metadata or {}
        #         key = (
        #             ("article", md.get("source"), md.get("article_id"))
        #             if md.get("article_id")
        #             else ("row", md.get("source"), md.get("idx"))
        #         )
        #         if key in seen:
        #             continue
        #         seen.add(key)
        #         uniq.append(d)
        #     docs = uniq
        # else:
        pairs = vdb.similarity_search_with_relevance_scores(
                query, k=k_retrieve
        )
        docs = as_docs(pairs)

        # ⭐ 最關鍵：用 cross-encoder 重新排序，只保留前 k 個
        docs = rerank_docs(query, docs, top_n=k)
        return docs

    return RunnableLambda(_retrieve).with_config({
        "run_name": "ChromaRetriever+Reranker",
        "tags": ["retriever", "chroma", "with-scores", "rerank"],
        "metadata": {"k": k}
    })

RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")  # 或 "cpu"

def build_chain():
    # 1) LLM
    llm = ChatOllama(
#        model="cwchang/llama-3-taiwan-8b-instruct:latest",
        model="qwen3:latest",
        temperature=0,
        # num_gpu=0,          # ⭐ 關掉 GPU，全部跑在 CPU
    ).with_config({
        "run_name": "Ollama-LLM",
        "tags": ["ollama", "tw-8b", "local"],
        "metadata": {"provider": "ollama"},
    })

    # 1.5) ✨ 新增：時間相關 query rewriter
    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一個查詢改寫器。\n"
            "現在時間：{now}（時區：Asia/Taipei）。\n"
            "目前學期：{acad_term}\n\n"
            "【參考知識】\n"
            "1. 民國紀年 = 西元年份 - 1911 (例：民國113年 = 2024年)。\n"
            "2. 「113學年度」約為 2024/08 ~ 2025/07。\n\n"
            "【任務目標】\n"
            "僅將使用者問題中的「相對時間詞」（如：今天、明天、這週、本學期、今年）替換為具體時間。\n"
            "**絕對不要**更動原本已經是「數字年份」的部分。\n\n"
            "【改寫範例】(請嚴格參考)\n"
            "- 輸入：「113上學期有什麼課？」 -> 輸出：「113上學期有什麼課？」(保持民國)\n"
            "- 輸入：「2024年行事曆」 -> 輸出：「2024年行事曆」(保持西元)\n"
            "- 輸入：「今年寒假什麼時候？」 -> 輸出：「2025年寒假什麼時候？」(將相對時間改寫為西元)\n\n"
            "- 輸入：「日本的姊妹校有哪些？」 -> 輸出：「日本的姊妹校有哪些？」(無時間詞，原樣輸出)\n\n"  # 🟢 關鍵修改：加入這個範例
            "【執行規則】\n"
            "1. 若問題中已有年份數字（無論民國或西元），**請原封不動保留，禁止進行換算**。\n"
            "2. 只有當問題中出現「今年、明年」等相對詞時，才使用「西元年」進行替換。\n"
            "3. 僅輸出改寫後的句子，不要任何解釋、不要標籤。\n"
        ),
        ("human", "{query}")
    ]).with_config({
        "run_name": "TemporalQueryRewriter",
        "tags": ["query-rewrite", "temporal"],
    })

    # 這條鏈輸入 {"query": str, "now": str}，輸出一個乾淨字串
    rewrite_chain = (rewrite_prompt | llm | StrOutputParser()).with_config({
        "run_name": "RewriteChain",
        "tags": ["chain", "rewrite"],
    })

    # 2) 提示詞

    # 在 build_chain 函式內修改 prompt
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #     "現在時間：{now}\n\n"
    #     "目前學期：{acad_term}\n\n"
    #     "學年等於民國紀年,114學年就是2025年。"
    #     "2025年8月1日~2026年7月31日都是114學年以此類推。\n"
    #     "你是大同大學資工系問答機器人。\n"
    #     "你會得到跟問題相關的文件以及系統提供的資訊（如現在時間、目前學期）,你只依據提供的文件內容回答問題,"
    #     "回答結尾需附上回答時參考資料的來源網址，若沒有則附上來源文件。"
    #     "若無法從文件中找到答案,請清楚說明。\n\n"
    #     "請以繁體中文作答，並使用 Markdown 格式來組織答案：\n"
    #     "- 使用 ## 二級標題來分隔不同主題\n"
    #     "- 使用列表 (- 或 1.) 來呈現多個項目\n"
    #     "- 使用 **粗體** 來強調重要資訊\n"
    #     "- 使用程式碼區塊 ```語言 來展示程式碼\n"
    #     "- 使用表格來呈現結構化資料\n\n"
    #     "{context}"),
    #     ("human", "{input}")
    # ]).with_config({
    #     "tags": ["chain", "stuff"],
    # })

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        # 🔥 補回這裡：讓模型知道現在幾點，才有辦法判斷「報名截止沒」
        "現在時間：{now}\n"
        "目前學期：{acad_term}\n\n"
        
        "你是大同大學資訊工程學系的問答助理。\n"
        "你的任務是根據下方的【參考文件】回答使用者的問題。\n\n"
        
        "【參考文件】\n"
        "###\n"
        "{context}\n"
        "###\n\n"

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

        # 🔥 關鍵新增：Few-Shot 範例，直接教它怎麼做 🔥
        "【回答範例 (Few-Shot)】\n"
        "Q: 資工系學生請假要怎麼請？\n"
        "A: 雖然文件中未特別提及資工系，但依據《大同大學學生請假規則》，學生請假需填寫請假單並檢附證明...\n"

        "【回答規則】\n"
        "1. **年份格式 (關鍵)**：涉及年份或學年度時，請務必優先使用「民國紀年」（例如：113學年、114年），嚴禁自動轉換為西元（如 2025年、2026年），除非文件中僅提供西元格式。這是為了符合校內規章慣例。\n"
        "2. **完整語句**：請以「主詞 + 謂語」的完整句子回答，語氣需肯定且專業。例如：「畢業門檻為128學分」，而非僅回答「128」。\n"
        "3. **綜合推論**：若答案散落在【參考文件】的不同段落，請嘗試將其整合。若資訊不完整但可合理推斷，請回答並加上「根據文件推測...」；不要因為沒有「逐字對應」就拒絕回答。\n"
        "4. **嚴禁瞎編**：若【參考文件】中完全沒有提及問題所需的關鍵資訊（特別是具體日期、人名、電話），請直接回答「文件中沒有相關資訊」，絕對不可自行生成數據。\n"
        "5. **多項條列**：若答案包含多個項目，請列出所有項目並用頓號或逗號隔開，保持語句通順。\n"
        "6. **禁止輸出思考**：請直接輸出最終答案，不要包含 `<think>` 標籤或推理過程。\n"
        ),
        ("human", "{input}")
    ])


    # # 在 build_chain 函式內修改 prompt
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #     "現在時間：{now}\n"
    #     "目前學期：{acad_term}\n\n"
    #     "你是大同大學資訊工程學系的問答機器人。\n"
    #     "你**只能**根據系統提供的資訊（現在時間、目前學期）以及下方提供的文件內容回答問題，請不要使用外部常識或自行推測。\n\n"
    #     "【回答規則】\n"
    #     "1. **第一句** 一定要用簡短、肯定、直白的語氣，直接回答問題的重點與數字。\n"
    #     "   - 例：『本系畢業需修滿 128 學分，其中系訂專業選修至少 26 學分。』\n"
    #     "2. 若需要補充說明（背景、條件、備註），請放在第二句之後，且保持精簡。\n"
    #     "3. 只要文件中有對應資訊，就**不要**說「資料未明確說明」「建議洽系辦」這類模糊或推託的句子。\n"
    #     "4. 只有在文件中完全找不到答案時，才這樣回答：\n"
    #     "   - 第一句直接寫：『在目前提供的文件中，找不到關於「{input}」的具體資訊，因此無法回答。』\n"
    #     "   - 然後可以用 1–2 句簡短說明可以向哪個單位詢問。\n"
    #     "5. 回答一律使用繁體中文。\n"
    #     "6. 在回答的結尾，列出本次實際參考到的文件或網址，格式如下（若沒有網址就寫文件名稱）：\n"
    #     "   參考資料：\n"
    #     "   - https://cse.ttu.edu.tw/... 或 檔案：xxx.pdf\n\n"
    #     "{context}"
    #     ),
    #     ("human", "{input}")
    # ]).with_config({
    #     "tags": ["chain", "stuff"],
    # })

    # 3) stuff chain
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt).with_config({
        "run_name": "StuffDocumentsChain",
        "tags": ["chain", "stuff"],
    })

    # 4) 向量庫 & 檢索器(含分數)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        # model_kwargs={"device": "cpu"},
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # 🔴 很推薦加
    )
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    # ✨ 關鍵：建立 scored_retriever，然後用 itemgetter 抽出 input 字串再餵檢索器
    scored_retriever = make_scored_retriever(vectordb, k=10) #　k=5　課本內容太多
    retriever_runnable = itemgetter("input") | scored_retriever  # dict -> str -> [Document]

    # 5) RAG 鏈（retriever + combine_docs）
    rag_chain = create_retrieval_chain(retriever_runnable, doc_chain).with_config({
        "run_name": "CampusRAG",
        "tags": ["campus-rag", "cli"],
    })

    # ✨ 改：回傳兩條鏈
    return {
        "rag": rag_chain,
        "rewrite": rewrite_chain,
    }

def pretty_print_snippets_with_scores(context_docs, max_chars: int = 240):
    seen = set()
    rows = []

    def dedup_key(d):
        md = d.metadata or {}
        src = md.get("source", "unknown")
        if md.get("page") is not None:
            return ("page", src, md.get("page"))
        if md.get("article_id") is not None:
            return ("article", src, md.get("article_id"))
        if md.get("idx") is not None:
            return ("row", src, md.get("idx"))
        return ("src", src)

    for d in context_docs:
        key = dedup_key(d)
        if key in seen:
            continue
        seen.add(key)

        md = d.metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page")
        title = md.get("title")
        chunk = md.get("chunk")
        ctype = md.get("content_type", md.get("type", "unknown"))

        display_idx = len(rows) + 1

        rel = md.get("relevance")
        rr  = md.get("rerank_score")
        rel_str = f"{float(rel):.3f}" if rel is not None else "—"
        rr_str  = f"{float(rr):.3f}" if rr is not None else "—"

        text = (d.page_content or "").replace("\n", " ").strip()
        snippet = (text[:max_chars] + "…") if len(text) > max_chars else text

        extra = ""
        if page is not None:
            extra = f"（第 {page} 頁）"
        elif title:
            extra = f"（{title}）"
        elif chunk is not None:
            extra = f"（chunk {chunk}）"

        header = f"{display_idx}. [{ctype}] {src}{extra}"
        rows.append(
            f"{header}\n"
            f"   └ 向量分數：{rel_str}｜rerank 分數：{rr_str}｜片段：{snippet}"
        )

    return "\n".join(rows)

def extract_clean_query(text: str) -> str:
    """從 rewriter 輸出中抽出『真正要拿去當 query 的那句話』"""
    if not text:
        return ""

    # 1) 如果 LLM 真的有照 <answer> 格式，可以優先抓 <answer>
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2) 把 <think>...</think> 整塊砍掉
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 3) 有時候會多行，我們取最後一行當 query（通常是那句問句）
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1]

@traceable(name="CLI-Ask", run_type="chain", metadata={"app": "campus_rag_cli"})
def ask(chains, q: str):
    """執行查詢：先用 LLM 做時間改寫，再丟給 RAG"""
    rag_chain = chains["rag"]
    rewrite_chain = chains["rewrite"]    # 取得台北時間

    now = datetime.now(ZoneInfo("Asia/Taipei"))
    roc_year = now.year - 1911
    today_roc = f"{roc_year}年{now.month}月{now.day}日"
    # now_time = now.strftime("%H:%M:%S")

    m, d = now.month, now.day

    # 依規定：8/1 開始新學年；2/1 開始第二學期
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

    # --- ✨ 新增：先讓 LLM 把 query 改寫成具體時間的問句 ---
    try:
        rewritten_q = rewrite_chain.invoke({
            "query": q,
            "now": now_str,
        }).strip()
        rewritten_q = extract_clean_query(rewritten_q)
    except Exception:
        # 避免 rewriter 掛掉整個系統，保底用原始問題
        rewritten_q = q

    if not rewritten_q:
        rewritten_q = q
    # ---------------------------------------------------------
    print(f"[DEBUG] rewritten query: {rewritten_q}")
    print(f"[DEBUG] cleaned rewrite: {rewritten_q!r}")

    # 之後就用改寫後的問題做檢索與回答
    return rag_chain.invoke({
        "input": rewritten_q,
        "now": now_str,
        "acad_term": acad_term,
        # 你也可以順便把原始/改寫後 query 傳進去，方便之後在 prompt 用
        "original_query": q,
        "rewritten_query": rewritten_q,
    })

if __name__ == "__main__":
    # 需要：export LANGSMITH_TRACING=true 與 LANGSMITH_API_KEY
    chains = build_chain() # 會拿到 {"rag": ..., "rewrite": ...}
    
    # 使用 rich 顯示歡迎訊息
    console.print(Panel.fit(
        "💬 大同大學資工系問答機器人\n輸入問題開始對話，按 Ctrl+C 結束",
        title="歡迎",
        border_style="cyan"
    ))
    
    try:
        while True:
            # 使用 rich 的 Prompt 取代 input
            console.print("[bold cyan]❓ 你的問題[/bold cyan]")
            q = Prompt.ask("")  # 空提示，讓使用者從空欄輸入
            
            if not q.strip():
                continue
            
            # ✨ 這裡改成丟 chains
            res = ask(chains, q)            
            raw = res["answer"]
            # --- 修改開始：使用 Regex 解析 XML ---
            thinking = ""
            answer = raw

            # 1. 嘗試提取 <think> 區塊
            think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()

            # 2. 嘗試提取 <answer> 區塊
            answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # 如果找不到 <answer> 標籤，可能模型沒跟隨格式
                # 為了保險，如果找到了 <think>，就把剩下的當作 answer
                # 或者直接顯示原始文字
                if think_match:
                    # 把 raw 中的 <think>...</think> 移除，剩下的當作回答
                    answer = raw.replace(think_match.group(0), "").strip()
            # --- 修改結束 ---

            # 印思考
            if thinking:
                console.print("\n[bold purple]🔍 思考過程：[/bold purple]")
                console.print(Panel(
                Markdown(thinking),
                border_style="purple",
                padding=(1,2)
            ))
            
            # 印最終回答 (如果有解析失敗，answer 會是原始全文，至少不會報錯)
            console.print("\n[bold green]✅ 最終回答：[/bold green]")
            console.print(Panel(
                Markdown(answer),
                border_style="green",
                padding=(1,2)
            ))
            
            # 顯示來源資訊
            console.print("\n[bold yellow]📚 參考來源：[/bold yellow]")
            sources_text = pretty_print_snippets_with_scores(res["context"])
            console.print(sources_text)
            
            console.print("[dim]" + "─" * 80 + "[/dim]\n")
            
    except KeyboardInterrupt:
        console.print("\n[bold blue]👋 再見！[/bold blue]")
