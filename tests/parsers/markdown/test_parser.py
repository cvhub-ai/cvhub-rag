from pathlib import Path
from textwrap import dedent

from src.assets.manager import AssetManager
from src.assets.resolver import AssetResolver
from src.models.enums import BlockType, ListType
from src.parsers.markdown.parser import MarkdownParser


def create_parser(tmp_path: Path) -> MarkdownParser:
    return MarkdownParser(
        asset_resolver=AssetResolver(),
        asset_manager=AssetManager(
            root_dir=tmp_path / "assets",
        ),
    )


def test_parse_heading_and_paragraph(tmp_path: Path) -> None:
    file_path = tmp_path / "test.md"

    file_path.write_text(
        dedent(
            """
            # Main Title

            Introduction text.

            ## Section

            Section text.
            """
        ).strip(),
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)
    blocks = document.pages[0].blocks

    assert len(blocks) == 4

    assert blocks[0].type == BlockType.HEADING
    assert blocks[0].text == "Main Title"
    assert blocks[0].structure.level == 1
    assert blocks[0].structure.parent_id is None

    assert blocks[1].type == BlockType.PARAGRAPH
    assert blocks[1].text == "Introduction text."
    assert blocks[1].structure.parent_id == blocks[0].id

    assert blocks[2].type == BlockType.HEADING
    assert blocks[2].text == "Section"
    assert blocks[2].structure.level == 2
    assert blocks[2].structure.parent_id == blocks[0].id

    assert blocks[3].type == BlockType.PARAGRAPH
    assert blocks[3].text == "Section text."
    assert blocks[3].structure.parent_id == blocks[2].id


def test_parse_nested_list(tmp_path: Path) -> None:
    file_path = tmp_path / "test.md"

    file_path.write_text(
        dedent(
            """
            # List

            - Parent
              - Child
            - Second
            """
        ).strip(),
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)

    list_blocks = [
        block for block in document.pages[0].blocks if block.type == BlockType.LIST_ITEM
    ]

    assert len(list_blocks) == 3

    parent = list_blocks[0]
    child = list_blocks[1]
    second = list_blocks[2]

    assert parent.text == "Parent"
    assert parent.list is not None
    assert parent.list.type == ListType.UNORDERED
    assert parent.list.level == 1

    assert child.text == "Child"
    assert child.list is not None
    assert child.list.level == 2
    assert child.structure.parent_id == parent.id

    assert second.text == "Second"
    assert second.list is not None
    assert second.list.level == 1


def test_parse_table(tmp_path: Path) -> None:
    file_path = tmp_path / "test.md"

    file_path.write_text(
        dedent(
            """
            | Method | Type |
            | --- | --- |
            | SIFT | Classical |
            | SuperPoint | Learned |
            """
        ).strip(),
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)

    table_blocks = [
        block for block in document.pages[0].blocks if block.type == BlockType.TABLE
    ]

    assert len(table_blocks) == 1

    table = table_blocks[0]
    assert table.table is not None
    assert table.table.rows == [
        ["Method", "Type"],
        ["SIFT", "Classical"],
        ["SuperPoint", "Learned"],
    ]


def test_parse_code(tmp_path: Path) -> None:
    file_path = tmp_path / "test.md"

    file_path.write_text(
        dedent(
            """
            ```python
            print("hello")
            ```
            """
        ).strip(),
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)

    blocks = document.pages[0].blocks

    assert len(blocks) == 1
    assert blocks[0].type == BlockType.CODE
    assert blocks[0].text == 'print("hello")'
    assert blocks[0].metadata["language"] == "python"


def test_parse_image_and_manage_asset(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake image")

    markdown_path = tmp_path / "test.md"
    markdown_path.write_text(
        dedent(
            """
            # Image

            ![Example](image.png)
            """
        ).strip(),
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(markdown_path)

    image_blocks = [
        block for block in document.pages[0].blocks if block.type == BlockType.IMAGE
    ]

    assert len(image_blocks) == 1

    block = image_blocks[0]
    assert block.image is not None
    assert block.image.caption == "Example"

    managed_path = tmp_path / "assets" / block.image.path
    assert managed_path.exists()
