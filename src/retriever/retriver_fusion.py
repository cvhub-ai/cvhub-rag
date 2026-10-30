from .base_retriever import BaseRetriever
from typing import List, Dict, Optional, Any
from ..dataclass.result import RetrieverResult, RetrieverChunk

class ResultFusion:
    '''
    
    '''


    def rrf_fusion(self, 
                    all_results: List[RetrieverResult], 
                     rrf_k: int = 60, 
                     top_k: int = 10) -> RetrieverResult:
        scores = {}
        docs = {}
        results_r = RetrieverResult(source="fusion")

        for results in all_results:
            for rank, r in enumerate(results.chunks):
                if r.chunk_id not in scores:
                    scores[r.chunk_id] = 0
                    docs[r.chunk_id] = r
                scores[r.doc_id] += 1 / (rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for chunk_id, _ in ranked[:top_k]:
            doc = docs[chunk_id]
            score = scores[chunk_id]
            chunk = RetrieverChunk(chunk_id=chunk_id, chunk_text=doc, score=score)
            results_r.chunks.append(chunk)
            
        return results_r

    def weighted_fusion(self, 
                         all_results: List[RetrieverResult], 
                         weights: List[float] = [0.5, 0.5], 
                         top_k:int = 10) -> RetrieverResult:
        scores = {}
        docs = {}
        for results, weight in zip(all_results, weights):
            for chunk in results.chunks:
                if chunk.chunk_id not in scores:
                    scores[chunk.chunk_id] = 0
                    docs[chunk.chunk_id] = chunk.chunk_text
                scores[chunk.chunk_id] += weight * chunk.score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = RetrieverResult(source="fusion")
        for chunk_id, score in ranked[:top_k]:
            doc = docs[chunk_id]
            chunk = RetrieverChunk(chunk_id, doc, score)
            results.append(chunk)
        return results
