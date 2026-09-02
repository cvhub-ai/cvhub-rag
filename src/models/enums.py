from enum import StrEnum


class FileType(StrEnum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    HTML = "html"
    TXT = "txt"


class ParserType(StrEnum):
    MARKDOWN = "markdown-parser"
    DOCLING = "docling"
    PADDLEOCR_VL = "paddleocr-vl"


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
