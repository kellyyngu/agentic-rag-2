import os
import json
import pickle
import re
import threading
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from rank_bm25 import BM25Okapi

from retrieval.document_processor import DocumentChunk


@dataclass
class IndexedDoc:
    chunk_id: str
    content: str
    source: str
    page: Optional[int]


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class BM25Index:
    """Thread-safe BM25 keyword index.

    A single RLock guards all mutations to ``self.docs`` and ``self.bm25`` so
    concurrent document uploads and searches never observe a partially-rebuilt
    index.  ``add()`` is CPU- and I/O-bound; callers inside an async context
    should dispatch it via ``asyncio.to_thread()``.
    """

    def __init__(self, data_dir: str = "/app/data"):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "bm25_index.pkl")
        self.docs_path = os.path.join(data_dir, "bm25_docs.json")
        self.bm25: Optional[BM25Okapi] = None
        self.docs: List[IndexedDoc] = []
        self._lock = threading.RLock()

    def load(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            try:
                with open(self.index_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                with open(self.docs_path, "r") as f:
                    raw = json.load(f)
                    self.docs = [IndexedDoc(**d) for d in raw]
                logger.info(f"[bm25] loaded index: {len(self.docs)} docs")
            except Exception as e:
                logger.warning(f"[bm25] load failed: {e}, starting fresh")
                self.bm25 = None
                self.docs = []

    def _rebuild(self):
        if not self.docs:
            self.bm25 = None
            return
        tokenized = [_tokenize(d.content) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)

    def add(self, chunks: List[DocumentChunk]):
        with self._lock:
            existing_ids = {d.chunk_id for d in self.docs}
            new_docs = [
                IndexedDoc(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    source=c.source,
                    page=c.page,
                )
                for c in chunks
                if c.chunk_id not in existing_ids
            ]
            if not new_docs:
                return

            self.docs.extend(new_docs)
            self._rebuild()
            self._save()
            logger.info(f"[bm25] added {len(new_docs)} docs, total={len(self.docs)}")

    def _save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.docs_path, "w") as f:
            json.dump([{"chunk_id": d.chunk_id, "content": d.content, "source": d.source, "page": d.page} for d in self.docs], f)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, str, Optional[str], Optional[int], float]]:
        """Returns list of (chunk_id, content, source, page, score)."""
        with self._lock:
            if not self.bm25 or not self.docs:
                return []

            tokens = _tokenize(query)
            scores = self.bm25.get_scores(tokens)

            # Get top-k indices
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

            results = []
            max_score = indexed[0][1] if indexed else 1.0
            for idx, score in indexed:
                if score <= 0:
                    continue
                doc = self.docs[idx]
                normalized = score / max(max_score, 1e-9)
                results.append((doc.chunk_id, doc.content, doc.source, doc.page, normalized))

            return results

    def clear(self):
        with self._lock:
            self.docs = []
            self.bm25 = None
            for p in [self.index_path, self.docs_path]:
                if os.path.exists(p):
                    os.remove(p)

    @property
    def doc_count(self) -> int:
        return len(self.docs)
