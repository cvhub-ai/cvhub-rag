from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from src.assets.manager import AssetManager
from src.assets.resolver import AssetResolver
from src.chunker.chunker import Chunker
from src.chunker.chunker_config import ChunkerConfig
from src.chunker.token_counter import TokenCounter
from src.models.chunk import ChunkType
from src.models.parsed_document import (
    Block,
    BlockType,
    FileType,
    ImageInfo,
    Page,
    ParsedDocument,
    ParserType,
    SourceInfo,
    StructureInfo,
    TableInfo,
)
from src.parsers.markdown.parser import MarkdownParser

TEST_DOCUMENT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "documents"
    / "test_computer_vision_markdown.md"
)


def test_chunker_with_markdown_document(tmp_path: Path):
    # Parse document
    asset_resolver = AssetResolver()
    asset_manager = AssetManager(root_dir=tmp_path / "assets")
    parser = MarkdownParser(asset_resolver=asset_resolver, asset_manager=asset_manager)
    document = parser.parse(TEST_DOCUMENT)

    # Create chunker
    config = ChunkerConfig()
    token_counter = TokenCounter()
    chunker = Chunker(config, token_counter)

    # Generate chunks
    chunks = chunker.chunk(document)

    # Validate results
    assert len(chunks) > 0
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.content.strip()
        assert chunk.token_count <= config.max_tokens
        assert chunk.token_count == token_counter.count(chunk.content)
        assert chunk.document_id == document.document_id
        assert chunk.source_block_ids
        assert chunk.page_start == 1
        assert chunk.page_end == 1

    # The fixture contains a Markdown table under section 7.1.
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == ChunkType.TABLE]
    assert table_chunks
    assert any(
        "Component | Responsibility | Example output" in chunk.content
        for chunk in table_chunks
    )
    assert all(
        chunk.section_title == "7.1 Example System Components" for chunk in table_chunks
    )
    assert any(chunk.chunk_type == ChunkType.TEXT for chunk in chunks)


class FakeTokenCounter:
    """Deterministic character-based tokenizer for unit tests."""

    special_tokens_count = 0

    def count(self, text: str) -> int:
        return len(text)

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(map(chr, tokens))


@pytest.fixture
def make_chunker():
    def factory(max_tokens: int = 120, overlap_tokens: int = 10) -> Chunker:
        config = ChunkerConfig(
            target_tokens=min(80, max_tokens),
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        return Chunker(
            config=config, token_counter=cast(TokenCounter, FakeTokenCounter())
        )

    return factory


def make_block(
    order: int,
    block_type: BlockType,
    text: str = "",
    *,
    level: int | None = None,
    table: TableInfo | None = None,
    image: ImageInfo | None = None,
    metadata: dict | None = None,
) -> Block:
    return Block(
        id=f"block_{order}",
        type=block_type,
        text=text,
        order=order,
        structure=StructureInfo(level=level),
        table=table,
        image=image,
        metadata=metadata or {},
    )


def make_document(*pages: list[Block]) -> ParsedDocument:
    return ParsedDocument(
        document_id="test-document",
        source=SourceInfo(
            file_name="test.md",
            file_type=FileType.MARKDOWN,
            parser=ParserType.MARKDOWN,
        ),
        pages=[
            Page(page_number=number, width=None, height=None, blocks=blocks)
            for number, blocks in enumerate(pages, start=1)
        ],
    )


def test_merges_text_blocks_and_preserves_source_ids(make_chunker):
    document = make_document(
        [
            make_block(2, BlockType.LIST_ITEM, "Second item"),
            make_block(1, BlockType.PARAGRAPH, "First paragraph"),
        ]
    )

    chunks = make_chunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.TEXT
    assert chunks[0].content == "First paragraph\n\nSecond item"
    assert chunks[0].source_block_ids == ["block_1", "block_2"]
    assert chunks[0].page_start == chunks[0].page_end == 1
    assert chunks[0].token_count == len(chunks[0].content)


def test_heading_hierarchy_and_boundaries(make_chunker):
    document = make_document(
        [
            make_block(1, BlockType.HEADING, "Chapter", level=1),
            make_block(2, BlockType.PARAGRAPH, "Introduction"),
            make_block(3, BlockType.HEADING, "Section A", level=2),
            make_block(4, BlockType.PARAGRAPH, "Section A body"),
            make_block(5, BlockType.HEADING, "Section B", level=2),
            make_block(6, BlockType.PARAGRAPH, "Section B body"),
        ]
    )

    chunks = make_chunker().chunk(document)

    assert [chunk.content for chunk in chunks] == [
        "Introduction",
        "Section A body",
        "Section B body",
    ]
    assert [chunk.metadata["section_path"] for chunk in chunks] == [
        ["Chapter"],
        ["Chapter", "Section A"],
        ["Chapter", "Section B"],
    ]
    assert [chunk.section_title for chunk in chunks] == [
        "Chapter",
        "Section A",
        "Section B",
    ]


def test_cross_page_chunk_and_max_tokens(make_chunker):
    document = make_document(
        [make_block(1, BlockType.PARAGRAPH, "Page one")],
        [make_block(2, BlockType.PARAGRAPH, "Page two")],
    )
    chunks = make_chunker(max_tokens=30).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == "Page one\n\nPage two"
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_flush_when_combination_exceeds_limit(make_chunker):
    document = make_document(
        [
            make_block(1, BlockType.PARAGRAPH, "A" * 25),
            make_block(2, BlockType.PARAGRAPH, "B" * 25),
        ]
    )

    chunks = make_chunker(max_tokens=30).chunk(document)

    assert [chunk.content for chunk in chunks] == ["A" * 25, "B" * 25]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_oversized_paragraph_is_split_with_overlap(make_chunker):
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    document = make_document([make_block(1, BlockType.PARAGRAPH, text)])

    chunks = make_chunker(max_tokens=12, overlap_tokens=3).chunk(document)

    assert len(chunks) > 1
    assert chunks[0].content == text[:12]
    assert chunks[1].content.startswith(text[9:12])
    assert all(chunk.token_count <= 12 for chunk in chunks)
    assert all(chunk.source_block_ids == ["block_1"] for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_table_creates_table_chunk_with_rows_metadata(make_chunker):
    rows = [["Name", "Value"], ["A", "1"], ["B", "2"]]
    document = make_document(
        [
            make_block(1, BlockType.TABLE, table=TableInfo(rows=rows)),
        ]
    )

    chunks = make_chunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.TABLE
    assert chunks[0].content == "Name | Value\nA | 1\nB | 2"
    assert chunks[0].metadata["table_rows"] == rows
    assert chunks[0].metadata["source_block_type"] == BlockType.TABLE.value


def test_long_table_splits_with_repeated_header(make_chunker):
    rows = [["Col", "Data"], ["a", "A" * 15], ["b", "B" * 15]]
    document = make_document(
        [
            make_block(1, BlockType.TABLE, table=TableInfo(rows=rows)),
        ]
    )

    chunks = make_chunker(max_tokens=30).chunk(document)

    assert len(chunks) == 2
    assert all(chunk.chunk_type == ChunkType.TABLE for chunk in chunks)
    assert all(chunk.content.startswith("Col | Data\n") for chunk in chunks)
    assert all(chunk.token_count <= 30 for chunk in chunks)


def test_image_keeps_reference_in_metadata(make_chunker):
    image = ImageInfo(path="assets/example.png", caption="Example figure")
    document = make_document(
        [
            make_block(1, BlockType.IMAGE, image=image),
        ]
    )

    chunks = make_chunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.IMAGE
    assert chunks[0].content == "Example figure"
    assert chunks[0].metadata["image_path"] == "assets/example.png"
    assert chunks[0].metadata["image_caption"] == "Example figure"


def test_code_keeps_language_and_is_standalone(make_chunker):
    document = make_document(
        [
            make_block(1, BlockType.PARAGRAPH, "Before code"),
            make_block(
                2, BlockType.CODE, "print('hello')", metadata={"language": "python"}
            ),
            make_block(3, BlockType.PARAGRAPH, "After code"),
        ]
    )

    chunks = make_chunker().chunk(document)

    assert [chunk.chunk_type for chunk in chunks] == [
        ChunkType.TEXT,
        ChunkType.CODE,
        ChunkType.TEXT,
    ]
    assert chunks[1].content == "print('hello')"
    assert chunks[1].metadata["language"] == "python"


@pytest.mark.parametrize(
    ("block_type", "expected_type"),
    [
        (BlockType.FORMULA, ChunkType.FORMULA),
        (BlockType.CAPTION, ChunkType.TEXT),
        (BlockType.HEADER, ChunkType.TEXT),
        (BlockType.FOOTER, ChunkType.TEXT),
        (BlockType.OTHER, ChunkType.TEXT),
    ],
)
def test_remaining_block_types(make_chunker, block_type, expected_type):
    document = make_document([make_block(1, block_type, "Example content")])

    chunks = make_chunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == expected_type
    assert chunks[0].content == "Example content"
    assert chunks[0].metadata["source_block_type"] == block_type.value


def test_empty_document_and_reusing_chunker(make_chunker):
    chunker = make_chunker()
    assert chunker.chunk(make_document([])) == []

    first = make_document(
        [
            make_block(1, BlockType.HEADING, "Old title", level=1),
            make_block(2, BlockType.PARAGRAPH, "First document"),
        ]
    )
    second = make_document([make_block(1, BlockType.PARAGRAPH, "Second document")])

    assert chunker.chunk(first)[0].section_title == "Old title"
    chunks = chunker.chunk(second)
    assert len(chunks) == 1
    assert chunks[0].section_title is None
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "Second document"
