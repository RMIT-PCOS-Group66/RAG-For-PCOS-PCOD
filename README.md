# pcos_rag_app (fastembed + FAISS, torch-free)

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Add PDFs
Put documents into:
```
data/pcos_docs/
```

## Run Streamlit UI
```bash
streamlit run ui/ui_app.py
```

## (Optional) Run API
```bash
python -m uvicorn app:app --reload --port 8000
```

## Evaluate
```bash
python eval/evaluate.py --data_dir ./data/pcos_docs
```
