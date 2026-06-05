"""
Hybrid retrieval engine.

Combines:
  * Dense retrieval  -> keepitreal/vietnamese-sbert cosine similarity
  * Lexical retrieval -> BM25 over pyvi-segmented tokens
Fused with Reciprocal Rank Fusion (RRF), then diversified with MMR.

The whole corpus is a single uploaded document, so brute-force numpy cosine
over a few hundred chunks is sub-millisecond on CPU — no ANN index needed,
and we avoid fragile native deps like faiss on Python 3.13.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import config
from embedder import Embedder
from textutils import chunk_text, tokenize_vi


@dataclass
class RetrievalResult:
    chunk: str
    score: float
    index: int


class HybridIndex:
    """Holds chunks + their dense vectors + BM25 model for one document."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self._bm25 = None
        self._tokenized: List[List[str]] = []
        self.ready: bool = False

    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #
    def build(self, text: str, doc_id: str = "none", use_cache: bool = True) -> int:
        """
        Chunk -> embed -> build BM25. Returns the number of chunks.
        
        Args:
            text: Document text
            doc_id: Document ID (for caching)
            use_cache: If True, try to load from cache; if False, always re-embed
        """
        from rank_bm25 import BM25Okapi
        import vector_cache

        # Try to load from cache first
        if use_cache:
            cached = vector_cache.load_cache(
                model_name=config.EMBED_MODEL_NAME,
                doc_id=doc_id,
                text=text,
            )
            
            if cached is not None:
                chunks, embeddings, metadata = cached
                print(f"[retriever] ✓ Loaded from cache (skip embedding)")
                
                # Build index from cached data
                with self._lock:
                    self.chunks = chunks
                    self.embeddings = embeddings
                    
                    self._tokenized = [tokenize_vi(c) for c in chunks]
                    self._tokenized = [t if t else ["<empty>"] for t in self._tokenized]
                    self._bm25 = BM25Okapi(self._tokenized)
                    
                    self.ready = True
                    print(f"[retriever] Indexed {len(chunks)} chunks from cache")
                    return len(chunks)

        # No cache or cache disabled -> embed from scratch
        from textutils import get_optimal_chunk_config
        
        if config.CHUNK_ADAPTIVE:
            chunk_config = get_optimal_chunk_config()
            print(f"[retriever] Using adaptive chunking: {chunk_config['chunk_size']} words, "
                  f"{chunk_config['overlap']} overlap ({chunk_config['overlap_ratio']:.0%})")
            print(f"[retriever] Reason: {chunk_config['reasoning']}")
        else:
            print(f"[retriever] Using manual config: {config.CHUNK_SIZE_WORDS} words, "
                  f"{config.CHUNK_OVERLAP_WORDS} overlap")
        
        chunks = chunk_text(text, use_adaptive=config.CHUNK_ADAPTIVE)
        if not chunks:
            chunks = [text.strip()] if text.strip() else []

        with self._lock:
            self.chunks = chunks
            if not chunks:
                self.embeddings = np.zeros((0, 1), dtype=np.float32)
                self._bm25 = None
                self._tokenized = []
                self.ready = False
                return 0

            emb = Embedder.get()
            # Encode chunks as passages (not queries)
            print(f"[retriever] Embedding {len(chunks)} chunks...")
            self.embeddings = emb.encode(chunks, is_query=False)

            self._tokenized = [tokenize_vi(c) for c in chunks]
            # guard against empty token lists (BM25 needs non-empty docs)
            self._tokenized = [t if t else ["<empty>"] for t in self._tokenized]
            self._bm25 = BM25Okapi(self._tokenized)

            self.ready = True
            print(f"[retriever] Indexed {len(chunks)} chunks")
            
            # Save to cache
            if use_cache:
                try:
                    vector_cache.save_cache(
                        model_name=config.EMBED_MODEL_NAME,
                        doc_id=doc_id,
                        text=text,
                        chunks=chunks,
                        embeddings=self.embeddings,
                    )
                except Exception as e:
                    print(f"[retriever] Warning: Failed to save cache: {e!r}")
            
            return len(chunks)

    # ------------------------------------------------------------------ #
    # Individual retrievers                                              #
    # ------------------------------------------------------------------ #
    def _dense(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self.embeddings.shape[0] == 0:
            return []
        sims = self.embeddings @ query_vec  # vectors are normalised -> cosine
        k = min(k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if self._bm25 is None or not self.chunks:
            return []
        tokens = tokenize_vi(query) or ["<empty>"]
        scores = self._bm25.get_scores(tokens)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    # ------------------------------------------------------------------ #
    # Fusion + diversification                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rrf(rankings: List[List[Tuple[int, float]]], k: int) -> Dict[int, float]:
        """Reciprocal Rank Fusion across multiple ranked lists."""
        fused: Dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _score) in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def _mmr(
        self,
        candidates: List[int],
        query_vec: np.ndarray,
        top_k: int,
        lambda_: float,
    ) -> List[int]:
        """Maximal Marginal Relevance re-ranking for diversity."""
        if not candidates:
            return []
        selected: List[int] = []
        remaining = list(candidates)
        cand_vecs = {i: self.embeddings[i] for i in candidates}
        rel = {i: float(cand_vecs[i] @ query_vec) for i in candidates}

        while remaining and len(selected) < top_k:
            best_idx, best_val = None, -1e9
            for i in remaining:
                if not selected:
                    val = rel[i]
                else:
                    div = max(float(cand_vecs[i] @ cand_vecs[j]) for j in selected)
                    val = lambda_ * rel[i] - (1 - lambda_) * div
                if val > best_val:
                    best_val, best_idx = val, i
            selected.append(best_idx)  # type: ignore[arg-type]
            remaining.remove(best_idx)  # type: ignore[arg-type]
        return selected

    # ------------------------------------------------------------------ #
    # Public search                                                      #
    # ------------------------------------------------------------------ #
    def search(
        self,
        queries: List[str],
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """
        Hybrid search over one or more query strings (e.g. the question stem
        plus each answer option). Results across queries are RRF-fused so the
        final context covers evidence relevant to every candidate answer.
        """
        with self._lock:
            if not self.ready or not self.chunks:
                return []

            top_k = top_k or config.TOP_K_CONTEXT
            emb = Embedder.get()

            rankings: List[List[Tuple[int, float]]] = []
            # Encode queries as queries (with proper prefix if needed)
            q_vecs = emb.encode(queries, is_query=True)
            for qi, q in enumerate(queries):
                rankings.append(self._dense(q_vecs[qi], config.TOP_K_DENSE))
                rankings.append(self._bm25_search(q, config.TOP_K_BM25))

            fused = self._rrf(rankings, config.RRF_K)
            if not fused:
                return []

            # Re-rank pool by fused RRF score, then diversify with MMR.
            # For small LLMs, favour relevance (higher lambda) over diversity.
            pool = sorted(fused.keys(), key=lambda i: -fused[i])
            pool = pool[: max(top_k * 4, config.TOP_K_DENSE + 4)]

            primary_vec = q_vecs[0]
            mmr_lambda = config.MMR_LAMBDA
            if len(queries) > 1:
                # Option-aware retrieval: keep chunks that match the stem best.
                mmr_lambda = min(0.85, mmr_lambda + 0.25)
            ordered = self._mmr(pool, primary_vec, top_k, mmr_lambda)

            return [
                RetrievalResult(chunk=self.chunks[i], score=fused[i], index=i)
                for i in ordered
            ]


# A single global index instance shared by the FastAPI app.
INDEX = HybridIndex()
