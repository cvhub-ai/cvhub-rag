from pathlib import Path
from statistics import median
from typing import Any, cast
from uuid import uuid4

import pymupdf

from src.models.enums import (
    BlockType,
    FileType,
    ParserType,
    StructureSource,
)
from src.models.parser_context import ParserContext
from src.models.parser_document import (
    Block,
    BoundingBox,
    Page,
    ParsedDocument,
    SourceInfo,
    StructureInfo,
)
from src.parsers.base import BaseDocumentParser


class PDFParser(BaseDocumentParser):
    def parse(
        self,
        file_path: str | Path,
    ) -> ParsedDocument:

        path = Path(file_path)
        context = ParserContext()

        pages: list[Page] = []

        with pymupdf.open(path) as pdf:
            page_data_list: list[dict[str, Any]] = [
                cast(
                    dict[str, Any],
                    page.get_text("dict", sort=True),
                )
                for page in pdf
            ]

            body_font_size = self._detect_body_font_size(page_data_list)

            heading_sizes = self._detect_heading_sizes(
                page_data_list,
                body_font_size,
            )

            for page_number, page_data in enumerate(
                page_data_list,
                start=1,
            ):
                page = self._parse_page(
                    page_data=page_data,
                    page_number=page_number,
                    context=context,
                    body_font_size=body_font_size,
                    heading_sizes=heading_sizes,
                )

                pages.append(page)

        return ParsedDocument(
            document_id=path.stem,
            source=SourceInfo(
                file_name=path.name,
                file_type=FileType.PDF,
                parser=ParserType.PDF,
            ),
            pages=pages,
        )

    def _parse_page(
        self,
        page_data: dict,
        page_number: int,
        context: ParserContext,
        body_font_size: float,
        heading_sizes: list[float],
    ) -> Page:
        blocks: list[Block] = []

        for raw_block in page_data["blocks"]:
            if raw_block.get("type") != 0:
                continue

            text = self._extract_block_text(raw_block)
            if not text:
                continue

            font_size = self._get_max_font_size(raw_block)

            block_type = self._detect_block_type(
                font_size=font_size,
                body_font_size=body_font_size,
            )

            heading_level: int | None = None

            if block_type == BlockType.HEADING:
                heading_level = self._resolve_heading_level(
                    font_size=font_size,
                    heading_sizes=heading_sizes,
                )

                parent_id = self._resolve_heading_parent(
                    context=context,
                    level=heading_level,
                )

                structure = StructureInfo(
                    parent_id=parent_id,
                    level=heading_level,
                    confidence=0.8,
                    source=StructureSource.RULE,
                )

            else:
                structure = StructureInfo(
                    parent_id=self._current_heading_id(context),
                    confidence=0.8,
                    source=StructureSource.RULE,
                )

            block = self._create_block(
                context=context,
                block_type=block_type,
                text=text,
                bbox=self._create_bbox(raw_block["bbox"]),
                structure=structure,
                metadata={
                    "font_size": font_size,
                },
            )

            blocks.append(block)

            if heading_level is not None:
                context.heading_stack.append(
                    (
                        heading_level,
                        block.id,
                    )
                )

        return Page(
            page_number=page_number,
            width=page_data.get("width"),
            height=page_data.get("height"),
            blocks=blocks,
        )

    @staticmethod
    def _extract_block_text(
        raw_block: dict,
    ) -> str:

        lines: list[str] = []

        for line in raw_block.get("lines", []):
            parts: list[str] = []

            for span in line.get("spans", []):
                text = span.get("text", "")

                if text:
                    parts.append(text)

            line_text = "".join(parts).strip()

            if line_text:
                lines.append(line_text)

        return "\n".join(lines)

    @staticmethod
    def _get_max_font_size(
        raw_block: dict,
    ) -> float:

        font_sizes: list[float] = []

        for line in raw_block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size")

                if size is not None:
                    font_sizes.append(float(size))

        if not font_sizes:
            return 0.0

        return max(font_sizes)

    @staticmethod
    def _detect_body_font_size(
        pages: list[dict],
    ) -> float:

        font_sizes: list[float] = []

        for page in pages:
            for block in page.get("blocks", []):
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size")

                        if text and size is not None:
                            font_sizes.append(float(size))

        if not font_sizes:
            return 0.0

        return median(font_sizes)

    @staticmethod
    def _detect_heading_sizes(
        pages: list[dict],
        body_font_size: float,
    ) -> list[float]:

        sizes: set[float] = set()

        for page in pages:
            for block in page.get("blocks", []):
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size")

                        if size is not None and float(size) >= body_font_size * 1.2:
                            sizes.add(round(float(size), 2))

        return sorted(
            sizes,
            reverse=True,
        )[:6]

    @staticmethod
    def _detect_block_type(
        font_size: float,
        body_font_size: float,
    ) -> BlockType:

        if body_font_size > 0 and font_size >= body_font_size * 1.2:
            return BlockType.HEADING

        return BlockType.PARAGRAPH

    @staticmethod
    def _resolve_heading_level(
        font_size: float,
        heading_sizes: list[float],
    ) -> int:

        rounded_size = round(font_size, 2)

        try:
            return heading_sizes.index(rounded_size) + 1
        except ValueError:
            return 1

    @staticmethod
    def _resolve_heading_parent(
        context: ParserContext,
        level: int,
    ) -> str | None:

        while context.heading_stack and context.heading_stack[-1][0] >= level:
            context.heading_stack.pop()

        if not context.heading_stack:
            return None

        return context.heading_stack[-1][1]

    @staticmethod
    def _current_heading_id(
        context: ParserContext,
    ) -> str | None:

        if not context.heading_stack:
            return None

        return context.heading_stack[-1][1]

    @staticmethod
    def _create_bbox(
        bbox: tuple | list,
    ) -> BoundingBox:

        return BoundingBox(
            x1=float(bbox[0]),
            y1=float(bbox[1]),
            x2=float(bbox[2]),
            y2=float(bbox[3]),
        )

    @staticmethod
    def _create_block(
        context: ParserContext,
        block_type: BlockType,
        text: str,
        bbox: BoundingBox,
        structure: StructureInfo,
        metadata: dict | None = None,
    ) -> Block:

        block = Block(
            id=f"block_{uuid4().hex}",
            type=block_type,
            text=text,
            order=context.next_order(),
            bbox=bbox,
            structure=structure,
            metadata=metadata or {},
        )

        context.blocks.append(block)

        return block
