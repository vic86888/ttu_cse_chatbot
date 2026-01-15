# eval_query_rewrite.py
import json
import re
from pathlib import Path

from dateutil import parser as dateparser

from query import build_chain, extract_clean_query

EVAL_QR_PATH = Path("eval/eval_query_rewrite.jsonl")

def load_eval_qr(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def extract_dates(text: str):
    """很簡單地用 regex 抓 YYYY年MM月DD日 或 YYYY-MM-DD 類型，再交給 dateparser"""
    if not text:
        return []

    candidates = re.findall(r"\d{4}[年/-]\d{1,2}[月/-]\d{1,2}", text)
    out = []
    for c in candidates:
        # 統一換成 2025-09-12 這種形式再 parse，偷懶寫法
        normalized = (
            c.replace("年", "-")
             .replace("月", "-")
             .replace("日", "")
        )
        try:
            dt = dateparser.parse(normalized, yearfirst=True, dayfirst=False)
            out.append(dt.date())
        except Exception:
            continue
    return out

def main():
    items = load_eval_qr(EVAL_QR_PATH)
    chains = build_chain()
    rewrite_chain = chains["rewrite"]

    total = 0
    exact_ok = 0
    loose_ok = 0

    for item in items:
        qid = item["id"]
        q = item["original_query"]
        now_str = item["now"]
        gold = item["gold_rewritten_query"]

        raw_pred = rewrite_chain.invoke({"query": q, "now": now_str})
        pred = extract_clean_query(raw_pred)

        gold_dates = extract_dates(gold)
        pred_dates = extract_dates(pred)

        total += 1

        if gold_dates and pred_dates and gold_dates == pred_dates:
            exact_ok += 1
            loose_ok += 1
            status = "✅ EXACT"
        elif gold_dates and pred_dates:
            # 只要日期集合有交集，就算 partial OK
            if set(gold_dates) & set(pred_dates):
                loose_ok += 1
                status = "🟡 PARTIAL"
            else:
                status = "❌ WRONG-DATE"
        else:
            status = "❌ NO-DATE"

        print(f"[{qid}] {status}")
        print(f"  原始：{q}")
        print(f"  now：{now_str}")
        print(f"  gold：{gold}")
        print(f"  pred：{pred}")
        print(f"  gold_dates={gold_dates}, pred_dates={pred_dates}")
        print("")

    print("=== Temporal Rewrite Eval ===")
    print(f"Total: {total}")
    print(f"Exact date match: {exact_ok}/{total} = {exact_ok/total:.3f}")
    print(f"Loose date overlap: {loose_ok}/{total} = {loose_ok/total:.3f}")

if __name__ == "__main__":
    main()
