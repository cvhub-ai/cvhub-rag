from .base_retriever import BaseRetriever
from typing import List, Dict, Optional, Any
from ..dataclass.result import RetrieverResult

class FusionRetriever(BaseRetriever):
    '''
    retriever(List[BaseRetriver]): a list of triever method \n
    method(str): fusion method, including reciprocal rank and weighted fusion \n
    weights(List(float)): the weight for weighted fusion \n
    rrf_k(int): ranking smoothing constant for reciprocal rank fusion \n
    name(str): name of the retriever \n
    top_k(int): the top_k results
    '''
    def __init__(self, 
                 retrievers: List[BaseRetriever],
                 method: str = "rrf",
                 weights: List[float] = None,
                 rrf_k:int = 60,
                 name: str = "fusion", 
                 top_k = 5):
        super().__init__(name, top_k)
        self.retrievers = retrievers
        self.method = method
        self.weights = weights or [1.0] * len(retrievers)
        self.rrf_k = rrf_k

    def _retrieve_impl(self, 
                       query: str, 
                       top_k: int, 
                       **kwargs) -> List[RetrieverResult]:
        all_results = []
        for retriever in self.retrievers:
            resp = retriever.retrieve(query, top_k*2)
            all_results.append(resp)

        if self.method == "rrf":
            return self._rrf_fusion(all_results, top_k)
        if self.method == "weighted":
            return self._weighted_fusion(all_results, top_k)
        return []

    def _rrf_fusion(self, 
                    all_results: List[List[RetrieverResult]], 
                    top_k: int) -> List[RetrieverResult]:
        scores = {}
        docs = {}

        for results in all_results:
            for rank, r in enumerate(results, start=1):
                if r.doc_id not in scores:
                    scores[r.doc_id] = 0
                    docs[r.doc_id] = r
                scores[r.doc_id] += 1 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:top_k]:
            doc = docs[doc_id]
            doc.score = score
            results.append(doc)
        return results

    def _weighted_fusion(self, 
                         all_results: List[List[RetrieverResult]], 
                         top_k: int) -> List[RetrieverResult]:
        scores = {}
        docs = {}
        for weight, results in zip(self.weights, all_results):
            for r in results:
                if r.doc_id not in scores:
                    scores[r.doc_id] = 0
                    docs[r.doc_id] = r
                scores[r.doc_id] += weight * r.score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:top_k]:
            doc = docs[doc_id]
            doc.score = score
            results.append(doc)
        return results