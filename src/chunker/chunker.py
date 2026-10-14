from collections.abc import Callable
from uuid import uuid4

from src.chunker.chunker_config import ChunkerConfig
from src.chunker.token_counter import TokenCounter
from src.models.chunk import Chunk, ChunkType
from src.models.parsed_document import Block, BlockType, ParsedDocument


class Chunker:
    _TEXT_TYPES = {
        BlockType.PARAGRAPH,
        BlockType.LIST_ITEM,
    }
    _STANDALONE_TYPES = {
        BlockType.TABLE: ChunkType.TABLE,
        BlockType.IMAGE: ChunkType.IMAGE,
        BlockType.CODE: ChunkType.CODE,
        BlockType.FORMULA: ChunkType.FORMULA,
        BlockType.CAPTION: ChunkType.TEXT,
        BlockType.HEADER: ChunkType.TEXT,
        BlockType.FOOTER: ChunkType.TEXT,
        BlockType.OTHER: ChunkType.TEXT,
    }

    def __init__(
        self,
        config: ChunkerConfig,
        token_counter: TokenCounter,
    ) -> None:
        self._config = config
        self._token_counter = token_counter
        self._heading_stack: list[tuple[int, str]] = []
        self._current_blocks: list[Block] = []
        self._block_pages: dict[str, int] = {}

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        self._heading_stack.clear()
        self._current_blocks.clear()
        self._block_pages.clear()

        def flush() -> None:
            if not self._current_blocks:
                return
            content = self._build_content(self._current_blocks)
            blocks = self._current_blocks.copy()
            self._current_blocks.clear()
            if content:
                chunks.append(
                    self._create_chunk(
                        document_id=document.document_id,
                        chunk_index=len(chunks),
                        content=content,
                        chunk_type=ChunkType.TEXT,
                        blocks=blocks,
                    )
                )

        for page in document.pages:
            for block in sorted(page.blocks, key=lambda item: item.order):
                self._block_pages[block.id] = page.page_number

                if block.type == BlockType.HEADING:
                    flush()
                    self._process_heading(block)
                    continue

                if block.type in self._TEXT_TYPES:
                    if not block.text.strip():
                        continue
                    self._add_text_block(block, flush, document.document_id, chunks)
                    continue

                flush()
                chunk_type = self._STANDALONE_TYPES.get(block.type, ChunkType.TEXT)

                if block.type == BlockType.TABLE and block.table and block.table.rows:
                    contents = self._split_table(block)
                else:
                    content = self._special_content(block)
                    contents = self._split_content(content) if content else []

                for content in contents:
                    metadata = self._special_metadata(block)
                    chunks.append(
                        self._create_chunk(
                            document_id=document.document_id,
                            chunk_index=len(chunks),
                            content=content,
                            chunk_type=chunk_type,
                            blocks=[block],
                            metadata=metadata,
                        )
                    )

        flush()
        return chunks

    def _add_text_block(
        self,
        block: Block,
        flush: Callable[[], None],
        document_id: str,
        chunks: list[Chunk],
    ) -> None:
        if self._token_counter.count(block.text.strip()) > self._config.max_tokens:
            flush()
            for content in self._split_content(block.text.strip()):
                chunks.append(
                    self._create_chunk(
                        document_id=document_id,
                        chunk_index=len(chunks),
                        content=content,
                        chunk_type=ChunkType.TEXT,
                        blocks=[block],
                    )
                )
            return

        candidate = self._build_content(self._current_blocks + [block])
        if self._token_counter.count(candidate) > self._config.max_tokens:
            flush()

        self._current_blocks.append(block)

    def _process_heading(self, block: Block) -> None:
        title = block.text.strip()
        if not title:
            return
        level = block.structure.level or 1
        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()
        self._heading_stack.append((level, title))

    @staticmethod
    def _build_content(blocks: list[Block]) -> str:
        return "\n\n".join(block.text.strip() for block in blocks if block.text.strip())

    def _split_content(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if self._token_counter.count(text) <= self._config.max_tokens:
            return [text]

        limit = self._config.max_tokens - self._token_counter.special_tokens_count
        overlap = self._config.overlap_tokens
        if limit <= 0:
            raise ValueError("max_tokens is too small")
        if not 0 <= overlap < limit:
            raise ValueError("Invalid overlap_tokens")

        tokens = self._token_counter.encode(text)
        results: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + limit, len(tokens))
            part = self._token_counter.decode(tokens[start:end]).strip()
            # Token decoding and re-encoding do not always preserve token counts.
            while part and self._token_counter.count(part) > self._config.max_tokens:
                end -= 1
                if end <= start:
                    raise ValueError("Cannot fit decoded text into max_tokens")
                part = self._token_counter.decode(tokens[start:end]).strip()
            if part:
                results.append(part)
            if end == len(tokens):
                break
            start = max(start + 1, end - overlap)
        return results

    def _split_table(self, block: Block) -> list[str]:
        assert block.table is not None
        rows = block.table.rows
        if not rows:
            return []

        header = " | ".join(rows[0])
        if self._token_counter.count(header) > self._config.max_tokens:
            return self._split_content(block.text)

        results: list[str] = []
        current = [header]
        for row in rows[1:]:
            row_text = " | ".join(row)
            candidate = "\n".join(current + [row_text])
            if self._token_counter.count(candidate) <= self._config.max_tokens:
                current.append(row_text)
                continue

            if len(current) > 1:
                results.append("\n".join(current))
                current = [header]

            if (
                self._token_counter.count(header + "\n" + row_text)
                <= self._config.max_tokens
            ):
                current.append(row_text)
            else:
                # An oversized row cannot be kept intact; split it as text.
                # Preserve the header as a separate searchable chunk.
                if not results or results[-1] != header:
                    results.append(header)
                results.extend(self._split_content(row_text))

        if len(current) > 1 or (not results and current):
            results.append("\n".join(current))
        return results

    @staticmethod
    def _special_content(block: Block) -> str:
        if block.type == BlockType.IMAGE:
            caption = block.image.caption if block.image else None
            path = block.image.path if block.image else None
            return (caption or block.text or path or "").strip()
        return block.text.strip()

    @staticmethod
    def _special_metadata(block: Block) -> dict:
        metadata = {"source_block_type": block.type.value}
        if block.type == BlockType.IMAGE and block.image is not None:
            metadata["image_path"] = block.image.path
            if block.image.caption:
                metadata["image_caption"] = block.image.caption
        if block.type == BlockType.CODE and block.metadata.get("language"):
            metadata["language"] = block.metadata["language"]
        if block.type == BlockType.TABLE and block.table is not None:
            metadata["table_rows"] = block.table.rows
        return metadata

    def _create_chunk(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        chunk_type: ChunkType,
        blocks: list[Block],
        metadata: dict | None = None,
    ) -> Chunk:
        token_count = self._token_counter.count(content)
        if token_count > self._config.max_tokens:
            raise ValueError(f"Chunk exceeds max_tokens: {token_count}")
        section_path = [title for _, title in self._heading_stack]
        page_numbers = [self._block_pages[block.id] for block in blocks]
        return Chunk(
            id=uuid4(),
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            chunk_type=chunk_type,
            page_start=min(page_numbers),
            page_end=max(page_numbers),
            section_title=section_path[-1] if section_path else None,
            source_block_ids=list(dict.fromkeys(block.id for block in blocks)),
            metadata={"section_path": section_path, **(metadata or {})},
        )
