import json
import re
import pandas as pd
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
    answer_correctness,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

# 假設您的 query.py 在同一目錄下
from query import build_chain, extract_clean_query
import backend.main_lamma as app_main

# =================設定區=================
EVAL_QA_PATH = Path("eval/test2.jsonl")  # 請確認路徑正確
run_config = RunConfig(
    timeout=600,    # 單題評分超時設定
    max_workers=1,  # 併發數
    max_retries=3,
)

# 修改正則表達式以捕捉內容 (Group 1)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
# =======================================

def parse_model_output(text: str):
    """
    解析模型輸出，分離思考過程與最終回答。
    回傳: (thinking_process, clean_answer)
    """
    if not isinstance(text, str):
        return "", text
    
    match = THINK_PATTERN.search(text)
    thinking_process = ""
    clean_answer = text

    if match:
        thinking_process = match.group(1).strip()
        # 將 <think>...</think> 整段移除，只留剩下的回答
        clean_answer = THINK_PATTERN.sub("", text).strip()
    else:
        # 如果沒有 <think> 標籤，假設整段都是回答，思考過程為空
        clean_answer = text.strip()
        
    return thinking_process, clean_answer

def load_eval_qa(path: Path):
    """讀取 jsonl 格式的測試題庫"""
    items = []
    if not path.exists():
        raise FileNotFoundError(f"找不到測試檔案: {path}")
        
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"跳過無效的 JSON 行: {line[:50]}...")
    return items

def get_now_and_term():
    """取得現在時間與學期資訊"""
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

def run_rag_for_eval(chains, question: str):
    """
    使用 main.py 的 ask 函式執行 RAG。
    注意：main.py 的 ask 會自動處理 rewrite 和 remove_thinking_tags
    """
    try:
        # 直接呼叫 API 的核心邏輯
        # 注意：main.ask 內部會自己計算 now 和 acad_term，所以不用傳進去
        res = app_main.ask(chains, question)
        
        # 取得乾淨的回答 (main.py 已經把 <think> 移除了)
        answer = res.get("answer", "")
        
        # 取得上下文
        context_docs = res.get("context", [])
        contexts_text = [d.page_content for d in context_docs]
        
        # 取得其他資訊
        rewritten_q = res.get("rewritten_query", "")
        

        # 🔥 直接從 result 拿思考過程 (如果 main.py 改好了)
        thinking_process = res.get("thinking_process", "")
        raw_answer = answer 

    except Exception as e:
        print(f"RAG Chain error: {e}")
        answer = f"Error: {str(e)}"
        raw_answer = str(e)
        thinking_process = ""
        contexts_text = []
        context_docs = []
        rewritten_q = ""

    return {
        "original_question": question,
        "rewritten_question": rewritten_q,
        "answer": answer,                 
        "thinking_process": thinking_process, 
        "raw_answer": raw_answer,         
        "contexts": contexts_text,
        "raw_context_docs": context_docs,
    }

def main():
    # 1. 載入題目
    eval_items = load_eval_qa(EVAL_QA_PATH)
    total_items = len(eval_items)
    print(f"載入 {total_items} 題測試題目")

    # ==========================================
    # 🔥 關鍵修改：手動初始化 main.py 的全域變數
    # ==========================================
    print("🔄 [Eval] 手動初始化 Reranker (模擬 API 啟動)...")
    # 強制設定為 GPU，確保跟 API 環境一致
    app_main.reranker = CrossEncoder(app_main.RERANK_MODEL_NAME, device="cuda")
    
    print("🔄 [Eval] 建立 RAG chains...")
    chains = app_main.build_chain()
    print("✅ 系統初始化完成！")
    # ==========================================

    rows = []
    print("=== 開始執行 RAG 生成 ===")
    
    # 3. 跑迴圈生成答案
    for i, item in enumerate(eval_items):
        q = item["question"]
        gt = item["expected_answer"]

        # 執行 RAG
        out = run_rag_for_eval(chains, q)

        rows.append({
            "question": q,
            "answer": out["answer"],
            "thinking_process": out["thinking_process"], 
            "contexts": out["contexts"],
            "ground_truth": gt,
            "original_question": out["original_question"],
            "rewritten_question": out["rewritten_question"],
            "raw_answer": out["raw_answer"], 
            "category": item.get("category", ""),
        })

        # 進度條
        current_count = i + 1
        if current_count % 5 == 0 or current_count == total_items:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 進度：{current_count} / {total_items}")

    # 4. 準備 RAGAS 評測資料集
    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"], # 評分時只給乾淨的回答
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])

    # 5. 設定評審模型
    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    # 6. 設定評審 Embeddings
    ragas_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("=== 開始進行 Ragas 評分 (這需要一點時間) ===")
    
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
    
    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness],
        llm=judge_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    # 7. 處理結果與檔案路徑建立
    df = result.to_pandas()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 建立資料夾結構： eval_result/rag_eval_{timestamp}/
    base_dir = Path("eval_result")
    output_dir = base_dir / f"rag_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[系統] 輸出目錄已建立: {output_dir}")

    # 合併分數
    final_data = []
    for i, row in df.iterrows():
        scores = {m: row.get(m, None) for m in metric_cols}
        item_data = rows[i].copy()
        item_data.update(scores)
        final_data.append(item_data)
    
    final_df = pd.DataFrame(final_data)

    print("\n=== RAG 評估完成 ===")

    # --- 存檔 1: CSV (Excel 用) ---
    csv_filename = output_dir / f"rag_eval_result_{timestamp}.csv"
    final_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"[已儲存] CSV 報表: {csv_filename}")

    # --- 存檔 2: JSON (詳細資料，已分離 thinking_process) ---
    json_filename = output_dir / f"rag_eval_detail_{timestamp}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"[已儲存] JSON 詳細資料: {json_filename}")

    # 共通門檻設定
    threshold = 0.6 

    # --- 存檔 3: TXT (低分檢討報告) ---
    bad_case_filename = output_dir / f"rag_bad_cases_report_{timestamp}.txt"
    
    with open(bad_case_filename, "w", encoding="utf-8") as f:
        f.write(f"=== RAG 評測檢討報告 ({timestamp}) ===\n\n")
        
        f.write("=== 平均分數 ===\n")
        for m in metric_cols:
            if m in final_df.columns:
                mean_score = final_df[m].mean()
                f.write(f"{m}: {mean_score:.3f}\n")
                print(f"{m}: {mean_score:.3f}")
        
        f.write("\n" + "="*50 + "\n")
        
        # 篩選任一指標低於門檻的
        bad_cases = final_df[
            (final_df['answer_relevancy'] < threshold) | 
            (final_df['faithfulness'] < threshold) |
            (final_df['answer_correctness'] < threshold) |
            (final_df['context_recall'] < threshold) 
        ]

        if not bad_cases.empty:
            f.write(f"\n=== 異常案例分析 (任一指標 < {threshold}) ===\n")
            for idx, row in bad_cases.iterrows():
                f.write(f"\n[索引 Q{idx}]\n")
                f.write(f"問題: {row.get('question', '')}\n")
                # 🔥 修改這一行，補上 Precision
                f.write(f"分數: Faith={row.get('faithfulness', 0):.2f} | Relevancy={row.get('answer_relevancy', 0):.2f} | Recall={row.get('context_recall', 0):.2f} | Precision={row.get('context_precision', 0):.2f} | Correctness={row.get('answer_correctness', 0):.2f}\n")
                f.write(f"模型回答: {row.get('answer', '')}\n")
                f.write(f"標準答案: {row.get('ground_truth', '')}\n")
                f.write("-" * 30 + "\n")
        else:
            f.write("\n恭喜！沒有發現分數過低的異常案例。\n")

    print(f"[已儲存] 低分檢討報告: {bad_case_filename}")

    # --- 存檔 4: TXT (優良案例報告 - 所有指標皆達標) ---
    high_quality_filename = output_dir / f"rag_high_quality_report_{timestamp}.txt"
    
    with open(high_quality_filename, "w", encoding="utf-8") as f:
        f.write(f"=== RAG 優良案例報告 ({timestamp}) ===\n")
        f.write(f"篩選標準: Faithfulness, Relevancy, Recall, Correctness 皆 >= {threshold}\n\n")
        
        # 🔥 修改邏輯：同時滿足四個指標 >= 0.6
        good_cases = final_df[
            (final_df['context_recall'].fillna(0) >= threshold) &
            (final_df['faithfulness'].fillna(0) >= threshold) &
            (final_df['answer_relevancy'].fillna(0) >= threshold) &
            (final_df['answer_correctness'].fillna(0) >= threshold)
        ]

        if not good_cases.empty:
            f.write(f"共發現 {len(good_cases)} 題全指標合格案例。\n")
            for idx, row in good_cases.iterrows():
                f.write(f"\n[索引 Q{idx}]\n")
                f.write(f"問題: {row.get('question', '')}\n")
                # 🔥 修改這一行，補上 Precision
                f.write(f"分數: Faith={row.get('faithfulness', 0):.2f} | Relevancy={row.get('answer_relevancy', 0):.2f} | Recall={row.get('context_recall', 0):.2f} | Precision={row.get('context_precision', 0):.2f} | Correctness={row.get('answer_correctness', 0):.2f}\n")
                f.write(f"模型回答: {row.get('answer', '')}\n")
                f.write(f"標準答案: {row.get('ground_truth', '')}\n")
                f.write("-" * 30 + "\n")
        else:
            f.write(f"\n無題目同時滿足四項指標 >= {threshold}。\n")

    print(f"[已儲存] 優良案例報告: {high_quality_filename}")

if __name__ == "__main__":
    main()