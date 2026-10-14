from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from src.models.chunk import Chunk, ChunkType


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def search_result_from_row(row: Mapping[str, Any]) -> SearchResult:
    metadata = dict(row["metadata"] or {})
    source_block_ids = metadata.pop("source_block_ids", [])

    chunk = Chunk(
        id=row["id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        token_count=row["token_count"],
        chunk_type=ChunkType(row["chunk_type"]),
        page_start=row["page_start"],
        page_end=row["page_end"],
        section_title=row["section_title"],
        source_block_ids=[UUID(str(block_id)) for block_id in source_block_ids],
        metadata=metadata,
    )

    return SearchResult(
        chunk=chunk,
        score=float(row["score"]),
    )
