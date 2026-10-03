from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class RetrieverResult:
    doc_id: str
    content: str
    score: Optional[float] = None
    results: Optional[EmbeddedDoc] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "results": self.results,
        }

@dataclass
class EmbeddedDoc:
    source: str
    result: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "result": self.result,
        }
    