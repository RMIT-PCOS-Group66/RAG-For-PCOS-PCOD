from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional
from engine import RAGFAISS
import pathlib
import asyncio

app = FastAPI(title="PCOS RAG (fastembed API)")
engine = RAGFAISS()

class IngestFolderRequest(BaseModel):
    data_dir: str = "./data/pcos_docs"
    glob: str = "*.pdf"

class FetchRequest(BaseModel):
    urls: Optional[List[str]] = None
    isbns: Optional[List[str]] = None
    kaggle_dataset: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    top_k: int = 4

class AskResponse(BaseModel):
    question: str
    supports: List[Dict]
    answer: str

@app.post("/ingest/folder")
def ingest_folder(req: IngestFolderRequest):
    d = pathlib.Path(req.data_dir)
    files = [str(p) for p in d.glob(req.glob)]
    engine.ingest_pdfs(files)
    return {"n_docs": len(files), "n_chunks": len(engine.texts)}

@app.post("/fetch")
def fetch(req: FetchRequest):
    async def _run():
        await engine.fetch_and_ingest(
            urls=req.urls or [],
            isbns=req.isbns or [],
            kaggle_dataset=req.kaggle_dataset,
        )
    asyncio.run(_run())
    return {"n_chunks": len(engine.texts)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    ans, supports = engine.answer(req.question, k=req.top_k)
    return AskResponse(question=req.question, supports=supports, answer=ans)
