from dataclasses import asdict, dataclass, field
from typing import Any

from .enums import (
    BlockType,
    FileType,
    ListType,
    ParserType,
    StructureSource,
)


@dataclass
class SourceInfo:
    file_name: str
    file_type: FileType
    parser: ParserType


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class StructureInfo:
    parent_id: str | None = None
    level: int | None = None
    confidence: float | None = None
    source: StructureSource = StructureSource.PARSER


@dataclass
class ListInfo:
    type: ListType
    level: int
    index: int | None = None
    marker: str | None = None


@dataclass
class TableInfo:
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class ImageInfo:
    path: str
    caption: str | None = None


@dataclass
class Block:
    id: str
    type: BlockType
    text: str
    order: int

    bbox: BoundingBox | None = None

    structure: StructureInfo = field(default_factory=StructureInfo)

    list: ListInfo | None = None
    table: TableInfo | None = None
    image: ImageInfo | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    page_number: int
    width: float | None
    height: float | None
    blocks: list[Block] = field(default_factory=list)


@dataclass
class UnifiedDocument:
    document_id: str
    source: SourceInfo
    pages: list[Page]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
