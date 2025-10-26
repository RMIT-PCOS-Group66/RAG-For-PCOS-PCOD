# engine.py
# RAG engine using fastembed + FAISS + BM25
# - No Kaggle / remote fetchers
# - Robust PDF extraction (PyMuPDF → pypdf fallback)
# - Small, RAM-friendly defaults

from __future__ import annotations

import re
import pathlib
from typing import List, Tuple, Dict, Optional

import numpy as np
import faiss

from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

# Robust PDF reading (PyMuPDF first, then pypdf)
import fitz  # PyMuPDF
from pypdf import PdfReader


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class RAGFAISS:
    """
    - Embeddings via fastembed TextEmbedding (cosine with IP on normalized vecs)
    - Hybrid retrieval: embeddings (alpha) + BM25 (1-alpha)
    - Pure local PDFs; no Kaggle/remote code
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 450,
        chunk_overlap: int = 80,
        alpha: float = 0.6,
        threshold: float = 0.35,         # similarity cutoff for "unanswered"
        embed_batch: int = 8,            # kept as attribute; used during encoding
    ):
        # Guard: if listing models fails, just try the default
        try:
            supported = set(TextEmbedding.list_supported_models())
        except Exception:
            supported = {"BAAI/bge-small-en-v1.5"}

        if model_name not in supported:
            fallback = (
                "BAAI/bge-small-en-v1.5"
                if "BAAI/bge-small-en-v1.5" in supported
                else next(iter(supported))
            )
            print(f"[INFO] '{model_name}' not supported; falling back to '{fallback}'")
            model_name = fallback

        self.embedder = TextEmbedding(model_name=model_name)
        self.model_name = model_name

        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.embed_batch = int(embed_batch)

        # corpus
        self.texts: List[str] = []
        self.metas: List[Dict] = []
        self.tokens: List[List[str]] = []

        # vector index
        self.X: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None

        # bm25
        self.bm25: Optional[BM25Okapi] = None

    # ---------- PDF reading ----------

    def _read_pdf(self, path: str) -> str:
        """Try PyMuPDF, then pypdf; raise on failure."""
        p = str(path)

        # PyMuPDF
        try:
            doc = fitz.open(p)
            pages = [pg.get_text() or "" for pg in doc]
            txt = "\n".join(pages).strip()
            if txt:
                return txt
        except Exception as e:
            print(f"[WARN] PyMuPDF failed on {p}: {e}")

        # pypdf
        try:
            rd = PdfReader(p)
            if getattr(rd, "is_encrypted", False):
                try:
                    rd.decrypt("")
                except Exception:
                    pass
            pages = [pg.extract_text() or "" for pg in rd.pages]
            txt = "\n".join(pages).strip()
            if txt:
                return txt
        except Exception as e:
            print(f"[WARN] pypdf failed on {p}: {e}")

        raise RuntimeError(
            f"Failed reading {p}. If it's image-only, please OCR or provide a text PDF."
        )

    # ---------- chunking ----------

    def _chunk(self, text: str) -> List[str]:
        sents = re.split(r"(?<=[\.\?\!])\s+", text.strip())
        if not sents:
            return []
        chunks, buf = [], ""
        for s in sents:
            if len(buf) + len(s) + 1 <= self.chunk_size:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)

        # simple overlap by repeating tail of previous chunk
        if self.chunk_overlap > 0 and len(chunks) > 1:
            merged = []
            for i, c in enumerate(chunks):
                if i == 0:
                    merged.append(c)
                else:
                    prev = merged[-1]
                    tail = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
                    merged.append((tail + " " + c).strip())
            chunks = merged
        return chunks

    # ---------- embeddings ----------

    def _encode(self, texts: List[str]) -> np.ndarray:
        # fastembed returns an iterator of normalized vectors
        vecs = list(self.embedder.embed(texts, batch_size=self.embed_batch, normalize=True))
        return np.asarray(vecs, dtype="float32")

    def _encode_query(self, query: str) -> np.ndarray:
        q = query
        name = (self.model_name or "").lower()

        # Prefer dedicated query encoder if available
        try:
            if hasattr(self.embedder, "query_embed"):
                vecs = list(self.embedder.query_embed([q], batch_size=1, normalize=True))
                return np.asarray(vecs, dtype="float32")
        except Exception:
            pass

        # Some models (e.g., e5) expect "query:" prefix; bge does not
        if "e5" in name:
            q = f"query: {q}"
        return self._encode([q])

    # ---------- ingest ----------

    def ingest_pdfs(self, file_paths: List[str]) -> None:
        texts, metas = [], []
        for fp in file_paths:
            try:
                raw = self._read_pdf(fp)
                parts = self._chunk(raw)
                for i, ch in enumerate(parts):
                    if ch.strip():
                        texts.append(ch)
                        metas.append({"source": str(fp), "chunk": i})
            except Exception as e:
                print(f"[WARN] Failed reading {fp}: {e}")

        if not texts:
            raise RuntimeError("No text extracted from provided PDFs.")

        # vectors
        X = self._encode(texts)
        self.X = X
        self.texts = texts
        self.metas = metas

        # FAISS IP index (cosine because vectors are normalized)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)

        # BM25
        self.tokens = [_norm(t).split() for t in texts]
        self.bm25 = BM25Okapi(self.tokens)

    # optional: add text blobs (not required for your UI)
    def ingest_text_blobs(self, blobs: List[str], source_tag: str = "blobs") -> None:
        new_texts, new_metas = [], []
        for i, blob in enumerate(blobs):
            for j, ch in enumerate(self._chunk(blob)):
                if ch.strip():
                    new_texts.append(ch)
                    new_metas.append({"source": source_tag, "item": i, "chunk": j})
        if not new_texts:
            return

        X_new = self._encode(new_texts)

        if self.index is None:
            self.texts = new_texts
            self.metas = new_metas
            self.X = X_new
            self.index = faiss.IndexFlatIP(X_new.shape[1])
            self.index.add(X_new)
            self.tokens = [_norm(t).split() for t in self.texts]
            self.bm25 = BM25Okapi(self.tokens)
            return

        # append
        self.texts.extend(new_texts)
        self.metas.extend(new_metas)
        self.index.add(X_new)
        self.X = np.vstack([self.X, X_new]) if self.X is not None else X_new
        self.tokens.extend([_norm(t).split() for t in new_texts])
        self.bm25 = BM25Okapi(self.tokens)

    # ---------- retrieval ----------

    def search(self, query: str, k: int = 4) -> Tuple[List[str], List[float], List[Dict]]:
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Index empty. Ingest PDFs first.")

        qv = self._encode_query(query)
        pool = min(k * 8, len(self.texts))
        D, I = self.index.search(qv, pool)

        emb_idxs = I[0].tolist()
        emb_scores = {i: float(s) for i, s in zip(emb_idxs, D[0].tolist())}

        # normalize embedding scores to [0,1]
        evec = np.zeros(len(self.texts), dtype="float32")
        for i, s in emb_scores.items():
            evec[i] = s
        if evec.max() > 0:
            evec = evec / max(1e-6, evec.max())

        # BM25 normalized to [0,1]
        bm = np.asarray(self.bm25.get_scores(_norm(query).split()), dtype="float32")
        if bm.max() > 0:
            bm = bm / max(1e-6, bm.max())

        combo = self.alpha * evec + (1.0 - self.alpha) * bm
        top = combo.argsort()[::-1][:k].tolist()

        texts = [self.texts[i] for i in top]
        sims = [float(combo[i]) for i in top]
        metas = [self.metas[i] for i in top]
        return texts, sims, metas

    # ---------- answer extraction ----------

    def _pick_sentences(self, passage: str, query: str, max_sents: int = 4) -> str:
        sents = re.split(r"(?<=[\.\?\!])\s+", passage.strip())
        if not sents:
            return ""
        X = self._encode(sents)
        qv = self._encode_query(query)
        scores = (X @ qv.T).ravel()

        # tiny bonus for lexical overlap
        q_terms = set(re.findall(r"\w+", query.lower()))
        bonus = np.array([0.1 * sum(1 for t in q_terms if t in s.lower()) for s in sents], dtype="float32")
        scores = scores + bonus

        idx = scores.argsort()[::-1][:max_sents]
        return " ".join([sents[i] for i in idx]).strip()

    def answer(self, query: str, k: int = 4) -> Tuple[str, List[Dict]]:
        texts, sims, _ = self.search(query, k=k)
        if not texts or (sims and max(sims) < self.threshold):
            return "unanswered", []
        ctx = "\n".join(texts[:2])
        extract = self._pick_sentences(ctx, query, max_sents=4)
        if not extract:
            return "unanswered", []
        supports = [{"text": t, "score": float(s)} for t, s in zip(texts, sims)]
        return extract, supports
