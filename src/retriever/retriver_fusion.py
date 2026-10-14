from .base_retriever import BaseRetriever
from typing import List, Dict, Optional, Any
from ..dataclass.result import RetrieverResult

class FusionRetriever:
    '''
    
    '''
    # def __init__(self):

    def rrf_fusion(self, 
                    all_results: List[List[RetrieverResult]], 
                    top_k: int, rrf_k: int = 60) -> List[RetrieverResult]:
        scores = {}
        docs = {}

        for results in all_results:
            for rank, r in enumerate(results, start=1):
                if r.doc_id not in scores:
                    scores[r.doc_id] = 0
                    docs[r.doc_id] = r
                scores[r.doc_id] += 1 / (rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:top_k]:
            doc = docs[doc_id]
            doc.score = score
            results.append(doc)
        return results

    def weighted_fusion(self, 
                         all_results: List[List[RetrieverResult]], 
                         top_k: int, weights: List[float]) -> List[RetrieverResult]:
        scores = {}
        docs = {}
        for weight, results in zip(weights, all_results):
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