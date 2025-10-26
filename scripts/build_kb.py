import json, pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine import RAGFAISS

def main():
    data_dir = PROJECT_ROOT / "data" / "pcos_docs"
    faq_path = PROJECT_ROOT / "data" / "pcos_faq" / "faq.json"

    pdfs = sorted([str(p) for p in data_dir.glob("*.pdf")])
    if not pdfs:
        raise SystemExit(f"No PDFs found in {data_dir}")

    rag = RAGFAISS(
        model_name="BAAI/bge-small-en-v1.5",  # or any supported model
        chunk_size=600,
        chunk_overlap=80,
        alpha=0.6,
        threshold=0.25,
    )

    print(f"Ingesting PDFs: {len(pdfs)}")
    rag.ingest_pdfs(pdfs)

    if faq_path.exists():
        faq = json.loads(faq_path.read_text(encoding="utf-8"))
        # We ingest only the answers as retrievable content (short, high-signal blobs)
        answers = [item["a"] for item in faq if item.get("a")]
        print(f"Ingesting Q&A answers: {len(answers)}")
        rag.ingest_text_blobs(answers, source_tag="faq")

    # Persist lightweight artifacts if you want (optional)
    out = PROJECT_ROOT / "data" / "cache" / "kb_stats.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"Total chunks: {len(rag.texts)}\nIndex dim: {rag.X.shape[1]}\n", encoding="utf-8")
    print(f"KB built. Chunks: {len(rag.texts)}")

if __name__ == "__main__":
    main()
