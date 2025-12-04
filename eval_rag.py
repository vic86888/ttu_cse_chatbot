# eval_rag.py
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,   # ← 新加這個
)

run_config = RunConfig(
    timeout=600,    # 單個打分工作的超時上限，先試 180 秒
    max_workers=1,  # 最多兩個評分任務同時跑，避免把 Ollama 壓爆
    max_retries=3,  # 超時時最多重試一次
)


from query import build_chain, extract_clean_query
from langchain_huggingface import HuggingFaceEmbeddings

# 🟢 請加入這一行：
from langchain_openai import ChatOpenAI

EVAL_QA_PATH = Path("eval/eval_qa_18 copy 2.jsonl")  # 你可以自己調整路徑

def load_eval_qa(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def run_rag_for_eval(chains, question: str, now_str: str, acad_term: str):
    """跟 query.ask 很像，但不印東西，乾淨回傳結果。"""
    rewrite_chain = chains["rewrite"]
    rag_chain = chains["rag"]

    # 1) 時間 query rewriter
    try:
        raw_rewrite = rewrite_chain.invoke({"query": question, "now": now_str})
        rewritten_q = extract_clean_query(raw_rewrite)
    except Exception:
        rewritten_q = question

    if not rewritten_q:
        rewritten_q = question

    # 2) RAG QA
    res = rag_chain.invoke({
        "input": rewritten_q,
        "now": now_str,
        "acad_term": acad_term,
        "original_query": question,
        "rewritten_query": rewritten_q,
    })

    answer = res["answer"]
    context_docs = res["context"]

    return {
        "original_question": question,
        "rewritten_question": rewritten_q,
        "answer": answer,
        "contexts": [d.page_content for d in context_docs],
        "raw_context_docs": context_docs,  # 如果你日後要看 metadata
    }

def get_now_and_term():
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    roc_year = now.year - 1911
    m, d = now.month, now.day

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
    return now_str, acad_term

def main():
    eval_items = load_eval_qa(EVAL_QA_PATH)
    print(f"載入 {len(eval_items)} 題測試題目")

    # 建 chain（會連同 embeddings, Chroma 一起載）
    chains = build_chain()
    now_str, acad_term = get_now_and_term()

    rows = []
    for item in eval_items:
        q = item["question"]
        gt = item["expected_answer"]

        out = run_rag_for_eval(chains, q, now_str, acad_term)

        rows.append({
            "question": q,
            "answer": out["answer"],
            "contexts": out["contexts"],
            "ground_truth": gt,
            "original_question": out["original_question"],
            "rewritten_question": out["rewritten_question"],
            "category": item.get("category", ""),
        })

    # 轉成 HuggingFace Dataset，給 ragas 用
    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])

    # 用哪個 LLM 當評審？你可以先用同一個 Qwen3，也可以換一個更穩定的雲端模型。

    # 使用 GPT-4o-mini (推薦，便宜又快)
    judge_llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        # openai_api_key="sk-...", # 建議讀取環境變數，不要寫死在 code 裡
    )

    # 2) 評審用 embeddings：直接用 BGE-M3（要 GPU 還是 CPU 你自己選）
    ragas_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        # 如果 GPU 還有空間，就：
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 開始評測
    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness], # 這樣就會多出一格
        llm=judge_llm,          # ✅ 雲端裁判 (負責打分邏輯)
        embeddings=ragas_embeddings, # ✅ 本地 Embeddings (負責算相似度)
        run_config=run_config,
    )

    print("=== RAG 評估結果 ===")

    # 1) 先轉成 DataFrame
    df = result.to_pandas()

    # 你目前用的四個指標名稱
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]

    # 2) 顯示每一題的分數
    print("\n=== 每題分數 ===")
    for i, row in df.iterrows():
        q = row.get("question", f"Q{i}")
        scores = "｜".join(
            f"{m}={row[m]:.3f}" if m in row and row[m] == row[m] else f"{m}=NaN"
            for m in metric_cols
        )
        print(f"- {q} -> {scores}")

    # 3) 顯示平均分數
    print("\n=== 平均分數 ===")
    for m in metric_cols:
        if m in df.columns:
            mean_val = df[m].mean()
            print(f"{m}: {mean_val:.3f}")

    # ⭐ 新增區塊：把 answer_relevancy 為 0 的問答印出來
    if "answer_relevancy" in df.columns:
        # 先找出哪幾個 index 的 answer_relevancy = 0
        zero_idx_list = df.index[df["answer_relevancy"] == 0].tolist()

        if zero_idx_list:
            print("\n=== answer_relevancy = 0 的題目（建議優先排查）=== ")
            for idx in zero_idx_list:
                # 用 idx 回頭拿你當初存好的 rows 裡的內容
                qa = rows[idx]   # rows 是你上面自己組的 list

                print("\n----------------------------------------")
                print(f"[索引] {idx}")
                print(f"[問題] {qa.get('question', '')}")
                print(f"[模型回答]\n{qa.get('answer', '')}")
                print(f"[標準答案 ground_truth]\n{qa.get('ground_truth', '')}")
                # 如果想順便看一下該題的分數，也可以印：
                print(f"[answer_relevancy 分數] {df.loc[idx, 'answer_relevancy']:.3f}")
        else:
            print("\n（沒有 answer_relevancy = 0 的題目）")

if __name__ == "__main__":
    main()
