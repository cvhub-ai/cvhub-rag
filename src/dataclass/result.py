from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class RetrieverChunk:
    chunk_id: UUID | str
    chunk_text: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk": self.chunk_text,
            "score": self.score
        }

@dataclass
class RetrieverResult:
    chunks: List[RetrieverChunk] = field(default_factory=list)
    source: Optional[str] = None

    # optional attributes 
    # middle_res: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "source": self.source,
            # "middle_res": self.middle_res,
        }