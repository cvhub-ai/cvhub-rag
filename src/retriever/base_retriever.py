from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..dataclass.result import RetrieverResult
import numpy as np

class BaseRetriever(ABC):

    def __init__(self,name: str = "", top_k: int = 10) -> None:
        self.name = name or self.__class__.__name__
        # self.top_k = top_k
        self.ready = False

    def retrieve(
        self, 
        query: str,
        top_k: int,
        **kwargs
        ) -> RetrieverResult:
        if not self.ready:
            self._ensure_ready()

        if not query or not query.strip():
            return []

        results = self._retrieve_impl(
            query=query.strip(),
            top_k=top_k,
            **kwargs
        )

        results = self._postprocess(results, top_k)

        return results

    def _ensure_ready(self):
        self.ready = True

    @abstractmethod
    def _retrieve_impl(
        self, 
        query: str, 
        top_k: int, 
        **kwargs
    ) -> RetrieverResult:
        raise NotImplementedError("Subclasses must implement the _retrieve_impl method.")

    def _postprocess(
        self, results: RetrieverResult, top_k: int
    ) -> RetrieverResult:
        chunks = results.chunks
        ranked_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        ranked_results = RetrieverResult(results.query_text, ranked_chunks[:top_k], results.source)
        return ranked_results

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        vecs_flat = vecs.ravel()
        min_vecs = np.min(vecs_flat)
        max_vecs = np.max(vecs_flat)
        vecs_norm = (vecs_flat - min_vecs) / (max_vecs - min_vecs)
        return vecs_norm

    def __repr__(self):
        return f"<{self.name} top_k={self.top_k} ready={self.ready}>"

