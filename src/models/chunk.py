from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    FORMULA = "formula"


@dataclass
class Chunk:
    id: UUID
    document_id: UUID
    chunk_index: int

    content: str
    token_count: int
    chunk_type: ChunkType

    page_start: int | None = None
    page_end: int | None = None

    section_title: str | None = None

    source_block_ids: list[UUID] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)
