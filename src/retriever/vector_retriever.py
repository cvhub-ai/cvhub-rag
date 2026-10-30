from ..dataclass.result import RetrieverResult, RetrieverChunk
from .base_retriever import BaseRetriever
from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict, Any, Optional
import numpy as np

class VectorRetriever(BaseRetriever):
    def __init__(
            self,
            vector_db: Any,
            embedded_model_path: str,
            name: str = "VectorRetriever",
            top_k: int = 10,
            score_threshold: float = 0.4
            ):
        super().__init__(name=name, top_k=top_k)
        self.embedded_model = BGEM3FlagModel(embedded_model_path, use_fp16=True)
        self.vector_db = vector_db
        self.score_threshold = score_threshold
        self.vector = np.empty((0, 1024))

    def _retrieve_impl(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> RetrieverResult:
        # 1. query vector
        query_vec = self.vectorize(query)

        # 2. vector search
        raw = self.vector_db.search(
            vector=query_vec,
            top_k=top_k,
        )
        
        # 3. post-process results
        results = RetrieverResult(
            # query_text=query,
            chunks=[],
            source="vector_retriever"
        )

        self.vector = np.empty((0, 1024))
        for item in raw:
            text_vec = np.array(item.vector)
            text_vec = text_vec.ravel()
            # text_vec = self.vectorize(text)
            self.vector = np.vstack((self.vector, text_vec))

        scores = self._score(query_vec, self.vector)
        for i in range(len(scores)):
            chunk = RetrieverChunk(raw[i].id, raw[i].content, scores[i])
            results.chunks.append(chunk)
        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        if vec1 is None or vec2 is None:
            return 0.0
        vec1 = vec1.ravel()
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2, axis=1)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def _score(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        cs = self._cosine_similarity(vec1, vec2)
        distance = np.pi - cs
        return self._normalize(distance)

    def vectorize(self, text: str) -> np.ndarray:
        embeddings = self.embedded_model.encode(text)
        res_vec = embeddings["dense_vecs"]
        vec = np.asarray(res_vec, dtype=np.float32)
        vec = vec/np.linalg.norm(vec)
        return vec
