from src.models.enums import BlockType
from src.models.parser_document import BoundingBox
from src.parsers.pdf.layout_region import PDFLayoutRegion
from src.parsers.pdf.parser import PDFParser


def create_parser() -> PDFParser:
    return PDFParser.__new__(PDFParser)


def test_create_paragraph_block() -> None:
    parser = create_parser()

    region = PDFLayoutRegion(
        label="paragraph",
        page_number=1,
        bbox=BoundingBox(
            x1=10,
            y1=20,
            x2=100,
            y2=50,
        ),
        text="Computer vision paragraph.",
    )

    block = parser._create_block_from_region(
        region,
        order=1,
    )

    assert block is not None
    assert block.type == BlockType.PARAGRAPH
    assert block.text == "Computer vision paragraph."
    assert block.order == 1
    assert block.bbox == region.bbox


def test_create_heading_block() -> None:
    parser = create_parser()

    region = PDFLayoutRegion(
        label="section_header",
        page_number=1,
        bbox=BoundingBox(
            x1=0,
            y1=0,
            x2=100,
            y2=20,
        ),
        text="Object Detection",
    )

    block = parser._create_block_from_region(
        region,
        order=1,
    )

    assert block is not None
    assert block.type == BlockType.HEADING
    assert block.text == "Object Detection"


def test_create_table_block() -> None:
    parser = create_parser()

    rows = [
        ["Method", "Type"],
        ["SIFT", "Classical"],
    ]

    region = PDFLayoutRegion(
        label="table",
        page_number=1,
        bbox=BoundingBox(
            x1=0,
            y1=0,
            x2=200,
            y2=100,
        ),
        text="Method | Type\nSIFT | Classical",
        table_rows=rows,
    )

    block = parser._create_block_from_region(
        region,
        order=1,
    )

    assert block is not None
    assert block.type == BlockType.TABLE
    assert block.table is not None
    assert block.table.rows == rows


def test_create_image_block() -> None:
    parser = create_parser()

    region = PDFLayoutRegion(
        label="picture",
        page_number=1,
        bbox=BoundingBox(
            x1=10,
            y1=10,
            x2=300,
            y2=200,
        ),
        image="document/images/image_123.png",
        caption="Architecture diagram",
    )

    block = parser._create_block_from_region(
        region,
        order=1,
    )

    assert block is not None
    assert block.type == BlockType.IMAGE
    assert block.image is not None
    assert block.image.path == "document/images/image_123.png"
    assert block.image.caption == "Architecture diagram"


def test_unknown_region_returns_none() -> None:
    parser = create_parser()

    region = PDFLayoutRegion(
        label="unknown_type",
        page_number=1,
        bbox=BoundingBox(
            x1=0,
            y1=0,
            x2=100,
            y2=100,
        ),
        text="Unknown",
    )

    block = parser._create_block_from_region(
        region,
        order=1,
    )

    assert block is None
