from ..dataclass.result import RetrieverResult, EmbeddedDoc
from .base_retriever import BaseRetriever
from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict, Any, Optional
import numpy as np

class VectorRetriever(BaseRetriever):
    def __init__(
            self,
            embedded_model_path: str,
            vector_db,
            name: str = "VectorRetriever",
            top_k: int = 5,
            score_threshold: float = 1.0
            ):
        super().__init__(name=name, top_k=top_k)
        self.embedded_model = BGEM3FlagModel(embedded_model_path, use_fp16=True)
        self.vector_db = vector_db
        self.score_threshold = score_threshold

    def _retrieve_impl(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> List[RetrieverResult]:
        # 1. query vector
        query_vec = self.vectorize(query)

        # 2. vector search
        raw = self.vector_db.search(
            vector=query_vec,
            top_k=top_k,
        )
        
        # 3. post-process results
        results = []
        for item in raw:
            score = item.get("score", 0.0)
            if score < self.score_threshold:
                continue
            res = EmbeddedDoc(source=self.name, result=item.get("result", {}))
            results.append(RetrieverResult(
                doc_id=str(item.get("id", "")),
                content=item.get("content", ""),
                score=score,
                results=res
            ))
        return results

    def vectorize(self, query: str) -> np.ndarray:
        embeddings = self.embedded_model.encode(query)
        res_vec = embeddings["dense_vecs"][0]
        vec = np.asarray(res_vec, dtype=np.float32)
        vec = vec/np.linalg.norm(vec)
        return vec
