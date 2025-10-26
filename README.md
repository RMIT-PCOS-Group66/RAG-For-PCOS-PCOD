# PCOS Nutrition Assistant (RAG System)
*A Retrieval-Augmented Generation prototype that provides evidence-based nutrition and lifestyle guidance for PCOS/PCOD.*

## Overview
This project implements a **test-driven RAG pipeline**: PDF ingestion → cleaning → 500-word chunks → **Sentence-BERT** embeddings → **FAISS** vector search → **LangChain** retrieval+generation → answer or **“I don’t know”** (abstention) for low confidence.

## Features
- Semantic retrieval with Sentence-BERT  
- Fast vector search via FAISS  
- Context-aware responses through LangChain (RAG)  
- Abstention mechanism to reduce hallucinations  
- Simple API/UI for interactive testing

## Project Structure
.
├─ app.py                # Flask API (if used)
├─ ui/                   # Streamlit UI (ui_app.py)
├─ engine.py             # core RAG functions
├─ scripts/              # utilities (index build, etc.)
├─ data/
│  └─ pcos_docs/         # put your PDFs here
├─ requirements.txt
└─ README.md


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
---

## How It Works (Quick)
1. Clean PDFs and chunk (~500 words, overlap)  
2. Encode chunks → **Sentence-BERT**  
3. Index embeddings → **FAISS**  
4. Embed query → retrieve top-k  
5. **LangChain** sends retrieved context to the LLM → grounded answer  
6. Low similarity → abstain (“I don’t know”)

---

## Team
- Felix George (S4077399)  
- Nithya Elsa John (S4077732)  
- Harshil Patel (S4106590)  
- Theresa Anitta Jaison (S4120039)

---

## Tech Stack
Python · LangChain · FAISS · Sentence-Transformers (Sentence-BERT) · Flask · Streamlit

---

## Academic Context
RMIT WIL Project: *Intelligent Nutrition & Lifestyle Support for PCOS/PCOD.*  
Functional prototype built and tested locally; demonstrates a Retrieval-Augmented Generation (RAG) pipeline with abstention for reliable evidence-based recommendations.

