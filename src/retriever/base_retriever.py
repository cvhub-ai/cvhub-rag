from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..dataclass.result import RetrieverResult

class BaseRetriever(ABC):

    def __init__(self,name: str = "", top_k: int = 5) -> None:
        self.name = name or self.__class__.__name__
        self.top_k = top_k
        self.ready = False

    def retrieve(
        self, 
        query: str,
        top_k: Optional[int] = None,
        **kwargs
        ) -> List[RetrieverResult]:
        if not self.ready:
            self._ensure_ready()

        if not query or not query.strip():
            return List()

        top_k = top_k or self.top_k

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
    ) -> List[RetrieverResult]:
        raise NotImplementedError("Subclasses must implement the _retrieve_impl method.")

    def _postprocess(
        self, results: List[RetrieverResult], top_k: int
    ) -> List[RetrieverResult]:
        results = sorted(results, key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def __repr__(self):
        return f"<{self.name} top_k={self.top_k} ready={self.ready}>"

