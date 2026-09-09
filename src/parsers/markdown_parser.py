from pathlib import Path
from uuid import uuid4

from markdown_it import MarkdownIt
from markdown_it.token import Token

from src.models.enums import (
    BlockType,
    FileType,
    ListType,
    ParserType,
    StructureSource,
)
from src.models.parser_context import ParserContext
from src.models.parser_document import (
    Block,
    ImageInfo,
    ListInfo,
    Page,
    ParsedDocument,
    SourceInfo,
    StructureInfo,
    TableInfo,
)
from src.parsers.base import BaseDocumentParser


class MarkdownParser(BaseDocumentParser):
    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark").enable("table")

    def parse(
        self,
        file_path: str | Path,
    ) -> ParsedDocument:

        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        context = ParserContext()

        tokens = self._markdown.parse(content)

        self._parse_tokens(tokens, context)

        page = Page(
            page_number=1,
            width=None,
            height=None,
            blocks=context.blocks,
        )

        return ParsedDocument(
            document_id=path.stem,
            source=SourceInfo(
                file_name=path.name,
                file_type=FileType.MARKDOWN,
                parser=ParserType.MARKDOWN,
            ),
            pages=[page],
        )

    def _parse_tokens(
        self,
        tokens: list[Token],
        context: ParserContext,
    ) -> None:

        index = 0

        while index < len(tokens):
            token = tokens[index]

            match token.type:
                case "heading_open":
                    index = self._parse_heading(tokens, index, context)

                case "paragraph_open":
                    index = self._parse_paragraph(tokens, index, context)

                case "bullet_list_open":
                    self._open_list(
                        context,
                        ListType.UNORDERED,
                    )

                case "ordered_list_open":
                    start = self._get_ordered_start(token)

                    self._open_list(
                        context,
                        ListType.ORDERED,
                        start=start,
                    )

                case "bullet_list_close" | "ordered_list_close":
                    self._close_list(context)

                case "list_item_open":
                    context.current_list_item_id = None

                case "list_item_close":
                    context.current_list_item_id = None

                case "fence" | "code_block":
                    self._parse_code(token, context)

                case "table_open":
                    index = self._parse_table(tokens, index, context)

            index += 1

    def _parse_heading(
        self,
        tokens: list[Token],
        index: int,
        context: ParserContext,
    ) -> int:

        open_token = tokens[index]

        level = int(open_token.tag.removeprefix("h"))

        inline_token = tokens[index + 1]
        text = self._extract_inline_text(inline_token)

        parent_id = self._resolve_heading_parent(
            context,
            level,
        )

        block = self._create_block(
            context=context,
            block_type=BlockType.HEADING,
            text=text,
            structure=StructureInfo(
                parent_id=parent_id,
                level=level,
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
        )

        context.heading_stack.append(
            (
                level,
                block.id,
            )
        )

        return index + 2

    def _parse_paragraph(
        self,
        tokens: list[Token],
        index: int,
        context: ParserContext,
    ) -> int:

        inline_token = tokens[index + 1]

        if context.list_stack:
            self._parse_list_item_content(
                inline_token,
                context,
            )
        else:
            self._parse_normal_paragraph(
                inline_token,
                context,
            )

        return index + 2

    def _parse_normal_paragraph(
        self,
        token: Token,
        context: ParserContext,
    ) -> None:

        images = self._extract_images(token)

        text = self._extract_inline_text(
            token,
            exclude_images=True,
        ).strip()

        if text:
            self._create_block(
                context=context,
                block_type=BlockType.PARAGRAPH,
                text=text,
                structure=StructureInfo(
                    parent_id=self._current_heading_id(context),
                    confidence=1.0,
                    source=StructureSource.MARKDOWN,
                ),
                metadata={"links": self._extract_links(token)},
            )

        for image in images:
            self._create_image_block(
                image,
                context,
            )

    def _parse_list_item_content(
        self,
        token: Token,
        context: ParserContext,
    ) -> None:

        list_context = context.list_stack[-1]

        text = self._extract_inline_text(
            token,
            exclude_images=True,
        ).strip()

        if not text:
            return

        list_context["current_index"] += 1

        if list_context["type"] == ListType.ORDERED:
            item_index = list_context["start"] + list_context["current_index"] - 1
            marker = f"{item_index}."
        else:
            item_index = None
            marker = "-"

        parent_id = self._resolve_list_parent(context)

        block = self._create_block(
            context=context,
            block_type=BlockType.LIST_ITEM,
            text=text,
            structure=StructureInfo(
                parent_id=parent_id,
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
            list_info=ListInfo(
                type=list_context["type"],
                level=len(context.list_stack),
                index=item_index,
                marker=marker,
            ),
            metadata={"links": self._extract_links(token)},
        )

        context.current_list_item_id = block.id
        list_context["last_item_id"] = block.id

    def _open_list(
        self,
        context: ParserContext,
        list_type: ListType,
        start: int = 1,
    ) -> None:

        parent_item_id = None

        if context.list_stack:
            parent_item_id = context.list_stack[-1].get("last_item_id")

        context.list_stack.append(
            {
                "type": list_type,
                "start": start,
                "current_index": 0,
                "last_item_id": None,
                "parent_item_id": parent_item_id,
            }
        )

    @staticmethod
    def _close_list(
        context: ParserContext,
    ) -> None:

        if context.list_stack:
            context.list_stack.pop()

    def _resolve_list_parent(
        self,
        context: ParserContext,
    ) -> str | None:

        current = context.list_stack[-1]

        parent_item_id = current.get("parent_item_id")

        if parent_item_id:
            return parent_item_id

        return self._current_heading_id(context)

    def _parse_code(
        self,
        token: Token,
        context: ParserContext,
    ) -> None:

        language = token.info.strip() or None

        self._create_block(
            context=context,
            block_type=BlockType.CODE,
            text=token.content.rstrip(),
            structure=StructureInfo(
                parent_id=self._current_heading_id(context),
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
            metadata={"language": language},
        )

    def _parse_table(
        self,
        tokens: list[Token],
        index: int,
        context: ParserContext,
    ) -> int:

        rows: list[list[str]] = []
        current_row: list[str] = []

        index += 1

        while index < len(tokens):
            token = tokens[index]

            if token.type == "table_close":
                break

            if token.type == "tr_open":
                current_row = []

            elif token.type == "tr_close":
                if current_row:
                    rows.append(current_row)

            elif (
                token.type in {"th_open", "td_open"}
                and index + 1 < len(tokens)
                and tokens[index + 1].type == "inline"
            ):
                current_row.append(self._extract_inline_text(tokens[index + 1]))

            index += 1

        text = "\n".join(" | ".join(row) for row in rows)

        self._create_block(
            context=context,
            block_type=BlockType.TABLE,
            text=text,
            structure=StructureInfo(
                parent_id=self._current_heading_id(context),
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
            table_info=TableInfo(rows=rows),
        )

        return index

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

    def _create_image_block(
        self,
        image: dict,
        context: ParserContext,
    ) -> None:

        self._create_block(
            context=context,
            block_type=BlockType.IMAGE,
            text=image["alt"],
            structure=StructureInfo(
                parent_id=self._current_heading_id(context),
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
            image_info=ImageInfo(
                path=image["path"],
                caption=image["alt"] or None,
            ),
        )

    @staticmethod
    def _create_block(
        context: ParserContext,
        block_type: BlockType,
        text: str,
        structure: StructureInfo,
        list_info: ListInfo | None = None,
        table_info: TableInfo | None = None,
        image_info: ImageInfo | None = None,
        metadata: dict | None = None,
    ) -> Block:

        block = Block(
            id=f"block_{uuid4().hex}",
            type=block_type,
            text=text,
            order=context.next_order(),
            bbox=None,
            structure=structure,
            list=list_info,
            table=table_info,
            image=image_info,
            metadata=metadata or {},
        )

        context.blocks.append(block)

        return block

    @staticmethod
    def _get_ordered_start(
        token: Token,
    ) -> int:

        start = token.attrGet("start")

        if start is None:
            return 1

        return int(start)

    @staticmethod
    def _extract_inline_text(
        token: Token,
        exclude_images: bool = False,
    ) -> str:

        if not token.children:
            return token.content

        parts: list[str] = []

        for child in token.children:
            if child.type in {"text", "code_inline"}:
                parts.append(child.content)

            elif child.type in {
                "softbreak",
                "hardbreak",
            }:
                parts.append("\n")

            elif child.type == "image" and not exclude_images:
                parts.append(child.content)

        return "".join(parts)

    @staticmethod
    def _extract_images(
        token: Token,
    ) -> list[dict]:

        if not token.children:
            return []

        images = []

        for child in token.children:
            if child.type != "image":
                continue

            images.append(
                {
                    "alt": child.content,
                    "path": child.attrGet("src") or "",
                }
            )

        return images

    @staticmethod
    def _extract_links(
        token: Token,
    ) -> list[dict]:

        if not token.children:
            return []

        links: list[dict] = []

        active_link: dict | None = None

        for child in token.children:
            if child.type == "link_open":
                active_link = {
                    "url": child.attrGet("href"),
                    "text": "",
                }

            elif child.type == "text" and active_link is not None:
                active_link["text"] += child.content

            elif child.type == "link_close" and active_link is not None:
                links.append(active_link)
                active_link = None

        return links
