from dataclasses import dataclass

from src.models.parser_document import BoundingBox


@dataclass
class PDFLayoutRegion:
    label: str
    page_number: int
    bbox: BoundingBox
    text: str | None = None
    table_rows: list[list[str]] | None = None
    image: str | None = None
    caption: str | None = None
