#!/usr/bin/env python3
# evaluate.py  — run Walert RAG on gold/ood and compute Precision/Recall/F1 on gold

import json, sys, pathlib as pl, re
from typing import Dict, List, Tuple
sys.path.insert(0, str(pl.Path(__file__).resolve().parents[1]))  # project root on path
from engine import RAGFAISS  # your existing engine

# ----------------------------
# Text normalization utilities
# ----------------------------
_punct_re = re.compile(r"[^a-z0-9\s]+")
_ws_re = re.compile(r"\s+")

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = _punct_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s

def tokenize(text: str) -> List[str]:
    return normalize(text).split()

# -------------------------------------------------
# Token-overlap Precision / Recall / F1 (bag-of-words)
# -------------------------------------------------
from collections import Counter

def prf_token_overlap(gold: str, pred: str) -> Tuple[float, float, float, int, int, int]:
    gold_toks = tokenize(gold)
    pred_toks = tokenize(pred)

    if not gold_toks and not pred_toks:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if not pred_toks:
        return 0.0, 0.0, 0.0, 0, len(gold_toks), 0
    if not gold_toks:
        return 0.0, 1.0, 0.0, len(pred_toks), 0, 0

    g = Counter(gold_toks)
    p = Counter(pred_toks)
    overlap = sum((g & p).values())

    precision = overlap / max(1, sum(p.values()))
    recall    = overlap / max(1, sum(g.values()))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, sum(p.values()), sum(g.values()), overlap

def exact_match_contains(gold: str, pred: str) -> int:
    # counts 1 if normalized gold is a substring of normalized pred, else 0
    g = normalize(gold)
    p = normalize(pred)
    if not g:
        return 0
    return 1 if g in p else 0

# ----------------------------
# Aggregate evaluation on gold
# ----------------------------
def evaluate_predictions(gold_items: List[Dict], predictions: List[Dict]) -> Dict:
    assert len(gold_items) == len(predictions), "gold/pred length mismatch"

    em_hits = 0
    P_sum = R_sum = F1_sum = 0.0
    n = len(gold_items)

    per_item = []
    for it, pr in zip(gold_items, predictions):
        gold = it.get("a", "")
        pred = pr.get("pred", "")
        em = exact_match_contains(gold, pred)
        P, R, F1, pred_len, gold_len, overlap = prf_token_overlap(gold, pred)

        em_hits += em
        P_sum += P
        R_sum += R
        F1_sum += F1

        per_item.append({
            "q": it.get("q", ""),
            "gold": gold,
            "pred": pred,
            "exact_match_contains": int(em),
            "precision_tokens": P,
            "recall_tokens": R,
            "f1_tokens": F1,
            "pred_token_count": pred_len,
            "gold_token_count": gold_len,
            "overlap_token_count": overlap,
        })

    report = {
        "n_items": n,
        "exact_match_rate": (em_hits / n) if n else 0.0,
        "avg_precision_tokens": (P_sum / n) if n else 0.0,
        "avg_recall_tokens":    (R_sum / n) if n else 0.0,
        "avg_f1_tokens":        (F1_sum / n) if n else 0.0,
        "items": per_item,
    }
    return report

# ----------------------------
# Main runner (unchanged + eval)
# ----------------------------
def main(
    data_dir: str = "./data/pcos_docs",
    gold: str = "./eval/gold.json",
    ood: str  = "./eval/ood.json",
    k: int = 4
):
    eng = RAGFAISS()
    pdfs = [str(p) for p in pl.Path(data_dir).glob("*.pdf")]
    if not pdfs:
        print(f"[WARN] No PDFs found under {data_dir}", file=sys.stderr)
    eng.ingest_pdfs(pdfs)

    # --- GOLD ---
    gold_items = json.load(open(gold, "r", encoding="utf-8"))
    out_gold = []
    for it in gold_items:
        q, a = it["q"], it["a"]
        pred, _ctx = eng.answer(q, k=k)
        out_gold.append({"q": q, "a": a, "pred": pred})

    # --- OOD ---
    ood_items = json.load(open(ood, "r", encoding="utf-8"))
    out_ood = []
    for q in ood_items:
        pred, _ctx = eng.answer(q, k=k)
        out_ood.append({"q": q, "pred": pred})

    # Persist predictions
    pl.Path("./eval/runs").mkdir(parents=True, exist_ok=True)
    gold_out_path = "./eval/runs/walert_gold_preds.json"
    ood_out_path  = "./eval/runs/walert_ood_preds.json"
    json.dump(out_gold, open(gold_out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(out_ood,  open(ood_out_path,  "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # --- EVALUATE on GOLD ---
    metrics = evaluate_predictions(gold_items, out_gold)

    # Save metrics
    metrics_path = "./eval/metrics.json"
    json.dump(metrics, open(metrics_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # Print a short console report
    print("\n📊 Evaluation (GOLD)")
    print(f"Items:                 {metrics['n_items']}")
    print(f"Exact-Match (contains): {metrics['exact_match_rate']:.3f}")
    print(f"Avg Precision (tokens): {metrics['avg_precision_tokens']:.3f}")
    print(f"Avg Recall    (tokens): {metrics['avg_recall_tokens']:.3f}")
    print(f"Avg F1        (tokens): {metrics['avg_f1_tokens']:.3f}")
    print(f"\nSaved predictions to:\n  {gold_out_path}\n  {ood_out_path}")
    print(f"Saved metrics to:\n  {metrics_path}\n")

if __name__ == "__main__":
    main()
