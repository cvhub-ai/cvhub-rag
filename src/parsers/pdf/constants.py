from src.models.enums import BlockType

OCR_TEXT_LABELS = frozenset(
    {
        "title",
        "section_header",
        "text",
        "paragraph",
        "list_item",
        "caption",
        "page_header",
        "page_footer",
        "footnote",
        "reference",
    }
)

DOCLING_BLOCK_TYPE_MAP: dict[str, BlockType] = {
    "title": BlockType.HEADING,
    "section_header": BlockType.HEADING,
    "text": BlockType.PARAGRAPH,
    "paragraph": BlockType.PARAGRAPH,
    "list_item": BlockType.LIST_ITEM,
    "table": BlockType.TABLE,
    "picture": BlockType.IMAGE,
    "caption": BlockType.CAPTION,
    "formula": BlockType.FORMULA,
    "code": BlockType.CODE,
    "page_header": BlockType.HEADER,
    "page_footer": BlockType.FOOTER,
}
