from pathlib import Path

from src.assets.manager import AssetManager
from src.assets.resolver import AssetResolver
from src.models.enums import BlockType, ListType
from src.parsers.html.parser import HtmlParser


def create_parser(tmp_path: Path) -> HtmlParser:
    return HtmlParser(
        asset_resolver=AssetResolver(),
        asset_manager=AssetManager(
            root_dir=tmp_path / "assets",
        ),
    )


def test_parse_heading_and_paragraph(tmp_path: Path) -> None:
    file_path = tmp_path / "test.html"

    file_path.write_text(
        """
            <html>
            <body>
            <h1>Main Title</h1>
            <p>Introduction.</p>
            <h2>Section</h2>
            <p>Section text.</p>
            </body>
            </html>
        """,
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)
    blocks = document.pages[0].blocks

    assert len(blocks) == 4

    assert blocks[0].type == BlockType.HEADING
    assert blocks[0].text == "Main Title"
    assert blocks[0].structure.level == 1

    assert blocks[1].type == BlockType.PARAGRAPH
    assert blocks[1].structure.parent_id == blocks[0].id

    assert blocks[2].type == BlockType.HEADING
    assert blocks[2].text == "Section"
    assert blocks[2].structure.parent_id == blocks[0].id

    assert blocks[3].type == BlockType.PARAGRAPH
    assert blocks[3].structure.parent_id == blocks[2].id


def test_parse_nested_list(tmp_path: Path) -> None:
    file_path = tmp_path / "test.html"

    file_path.write_text(
        """
            <html>
            <body>
            <ul>
                <li>
                    Parent
                    <ul>
                        <li>Child</li>
                    </ul>
                </li>
                <li>Second</li>
            </ul>
            </body>
            </html>
        """,
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)

    blocks = [
        block for block in document.pages[0].blocks if block.type == BlockType.LIST_ITEM
    ]

    assert len(blocks) == 3

    parent = blocks[0]
    child = blocks[1]
    second = blocks[2]

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
    file_path = tmp_path / "test.html"

    file_path.write_text(
        """
            <table>
                <tr>
                    <th>Method</th>
                    <th>Type</th>
                </tr>
                <tr>
                    <td>SIFT</td>
                    <td>Classical</td>
                </tr>
            </table>
        """,
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)

    blocks = document.pages[0].blocks

    assert len(blocks) == 1
    assert blocks[0].type == BlockType.TABLE
    assert blocks[0].table is not None
    assert blocks[0].table.rows == [
        ["Method", "Type"],
        ["SIFT", "Classical"],
    ]


def test_parse_code(tmp_path: Path) -> None:
    file_path = tmp_path / "test.html"

    file_path.write_text(
        """
            <pre><code class="language-python">print("hello")</code></pre>
        """,
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)

    blocks = document.pages[0].blocks

    assert len(blocks) == 1
    assert blocks[0].type == BlockType.CODE
    assert blocks[0].text == 'print("hello")'
    assert blocks[0].metadata["language"] == "python"


def test_parse_link_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "test.html"

    file_path.write_text(
        """
            <p>
                Read
                <a href="https://opencv.org/">OpenCV</a>.
            </p>
        """,
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)

    block = document.pages[0].blocks[0]

    assert block.type == BlockType.PARAGRAPH
    assert block.metadata["links"] == [
        {
            "url": "https://opencv.org/",
            "text": "OpenCV",
        }
    ]


def test_ignore_noise_tags(tmp_path: Path) -> None:
    file_path = tmp_path / "test.html"

    file_path.write_text(
        """
        <html>
            <body>
                <script>
                    console.log("should not appear");
                </script>
                <style>
                    body { color: red; }
                </style>
                <nav>
                    Navigation text
                </nav>
                <p>Real content.</p>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(file_path)

    blocks = document.pages[0].blocks

    assert len(blocks) == 1
    assert blocks[0].text == "Real content."


def test_parse_image_and_manage_asset(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake image")

    html_path = tmp_path / "test.html"
    html_path.write_text(
        """
            <html>
            <body>
            <img src="image.png" alt="Test image">
            </body>
            </html>
        """,
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(html_path)

    image_blocks = [
        block for block in document.pages[0].blocks if block.type == BlockType.IMAGE
    ]

    assert len(image_blocks) == 1

    block = image_blocks[0]
    assert block.image is not None
    assert block.image.caption == "Test image"

    managed_path = tmp_path / "assets" / block.image.path
    assert managed_path.exists()


def test_image_inside_list_item(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake image")

    html_path = tmp_path / "test.html"
    html_path.write_text(
        """
            <ul>
                <li>
                    Detection result
                    <img src="image.png" alt="Result">
                </li>
            </ul>
        """,
        encoding="utf-8",
    )

    parser = create_parser(tmp_path)
    document = parser.parse(html_path)

    blocks = document.pages[0].blocks

    list_block = next(block for block in blocks if block.type == BlockType.LIST_ITEM)

    image_block = next(block for block in blocks if block.type == BlockType.IMAGE)

    assert image_block.structure.parent_id == list_block.id
