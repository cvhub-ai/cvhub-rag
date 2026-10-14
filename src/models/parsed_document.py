from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FileType(StrEnum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    HTML = "html"
    TXT = "txt"


class ParserType(StrEnum):
    MARKDOWN = "markdown-parser"
    PDF = "pdf-parser"
    HTML = "html-parser"


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    FORMULA = "formula"
    CODE = "code"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


class StructureSource(StrEnum):
    MARKDOWN = "markdown"
    PARSER = "parser"
    STRUCTURE_MODEL = "structure-model"
    RULE = "rule"
    LLM = "llm"


class ListType(StrEnum):
    ORDERED = "ordered"
    UNORDERED = "unordered"


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
class ParsedDocument:
    document_id: str
    source: SourceInfo
    pages: list[Page]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
