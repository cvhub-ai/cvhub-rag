from typing import List, Any, Tuple
from FlagEmbedding import FlagReranker
from src.result import RetrieverChunk, RetrieverResult
import logging

logger = logging.getLogger(__name__)

class BGEReranker:
    def __init__(self, model_path: str, use_fp16: bool = True):
        logger.info(f"loading reranker model: {model_path}")
        self.reranker = FlagReranker(model_path, use_fp16=use_fp16)
        logger.info("reranker model loaded successfully")

    def rerank(
        self,
        query: str,
        chunks: RetrieverResult,
        top_k: int = 10,
    ) -> List[RetrieverChunk]:
        
        if not chunks:
            return []

        pairs = []
        valid_chunks = []
        for chunk in chunks.chunks:
            try:
                text = chunk.chunk_text
                if text and text.strip():
                    pairs.append([query, text])
                    valid_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")

        if not pairs:
            return []

        scores = self.reranker.compute_score(pairs, normalize=True)

        ranked = sorted(
            zip(valid_chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        ranked = [chunk for chunk, _ in ranked[:top_k]]

        return ranked