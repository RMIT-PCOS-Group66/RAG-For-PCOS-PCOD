import json, sys, pathlib as pl
sys.path.insert(0, str(pl.Path(__file__).resolve().parents[1]))  # project root on path
from engine import RAGFAISS

def main(data_dir="./data/pcos_docs", gold="./eval/gold.json", ood="./eval/ood.json", k=4):
    eng = RAGFAISS()
    pdfs = [str(p) for p in pl.Path(data_dir).glob("*.pdf")]
    eng.ingest_pdfs(pdfs)

    gold_items = json.load(open(gold, "r", encoding="utf-8"))
    out_gold = []
    for it in gold_items:
        q, a = it["q"], it["a"]
        pred, _ = eng.answer(q, k=k)
        out_gold.append({"q": q, "a": a, "pred": pred})

    ood_items = json.load(open(ood, "r", encoding="utf-8"))
    out_ood = []
    for q in ood_items:
        pred, _ = eng.answer(q, k=k)
        out_ood.append({"q": q, "pred": pred})

    pl.Path("./eval/runs").mkdir(parents=True, exist_ok=True)
    json.dump(out_gold, open("./eval/runs/walert_gold_preds.json", "w", encoding="utf-8"), indent=2)
    json.dump(out_ood, open("./eval/runs/walert_ood_preds.json", "w", encoding="utf-8"), indent=2)

if __name__ == "__main__":
    main()
