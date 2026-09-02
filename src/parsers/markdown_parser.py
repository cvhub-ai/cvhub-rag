from pathlib import Path
from uuid import uuid4

from markdown_it import MarkdownIt
from markdown_it.token import Token

from src.models.document import (
    Block,
    ImageInfo,
    ListInfo,
    Page,
    SourceInfo,
    StructureInfo,
    TableInfo,
    UnifiedDocument,
)
from src.models.enums import (
    BlockType,
    FileType,
    ListType,
    ParserType,
    StructureSource,
)
from src.parsers.base import BaseDocumentParser


class MarkdownParser(BaseDocumentParser):
    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark").enable("table")

        self._blocks: list[Block] = []

        self._order = 0

        self._heading_stack: list[tuple[int, str]] = []

        self._list_stack: list[dict] = []

        self._current_list_item_id: str | None = None

    def parse(
        self,
        file_path: str | Path,
    ) -> UnifiedDocument:

        path = Path(file_path)

        content = path.read_text(encoding="utf-8")

        self._reset()

        tokens = self._markdown.parse(content)

        self._parse_tokens(tokens)

        page = Page(
            page_number=1,
            width=None,
            height=None,
            blocks=self._blocks,
        )

        return UnifiedDocument(
            document_id=path.stem,
            source=SourceInfo(
                file_name=path.name,
                file_type=FileType.MARKDOWN,
                parser=ParserType.MARKDOWN,
            ),
            pages=[page],
        )

    def _reset(self) -> None:
        self._blocks = []
        self._order = 0
        self._heading_stack = []
        self._list_stack = []
        self._current_list_item_id = None

    def _parse_tokens(
        self,
        tokens: list[Token],
    ) -> None:

        print(tokens)
        index = 0

        while index < len(tokens):
            token = tokens[index]

            match token.type:
                case "heading_open":
                    index = self._parse_heading(
                        tokens,
                        index,
                    )

                case "paragraph_open":
                    index = self._parse_paragraph(
                        tokens,
                        index,
                    )

                case "bullet_list_open":
                    self._open_list(ListType.UNORDERED)

                case "ordered_list_open":
                    start = self._get_ordered_start(token)

                    self._open_list(
                        ListType.ORDERED,
                        start=start,
                    )

                case "bullet_list_close" | "ordered_list_close":
                    self._close_list()

                case "list_item_open":
                    self._current_list_item_id = None

                case "list_item_close":
                    self._current_list_item_id = None

                case "fence":
                    self._parse_code(token)

                case "code_block":
                    self._parse_code(token)

                case "table_open":
                    index = self._parse_table(
                        tokens,
                        index,
                    )

            index += 1

    def _parse_heading(
        self,
        tokens: list[Token],
        index: int,
    ) -> int:

        open_token = tokens[index]

        level = int(open_token.tag.removeprefix("h"))

        inline_token = tokens[index + 1]

        text = self._extract_inline_text(inline_token)

        parent_id = self._resolve_heading_parent(level)

        block = self._create_block(
            block_type=BlockType.HEADING,
            text=text,
            structure=StructureInfo(
                parent_id=parent_id,
                level=level,
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
        )

        self._heading_stack.append(
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
    ) -> int:

        inline_token = tokens[index + 1]

        if self._list_stack:
            self._parse_list_item_content(inline_token)
        else:
            self._parse_normal_paragraph(inline_token)

        return index + 2

    def _parse_normal_paragraph(
        self,
        token: Token,
    ) -> None:

        images = self._extract_images(token)

        text = self._extract_inline_text(
            token,
            exclude_images=True,
        ).strip()

        if text:
            self._create_block(
                block_type=BlockType.PARAGRAPH,
                text=text,
                structure=StructureInfo(
                    parent_id=self._current_heading_id(),
                    confidence=1.0,
                    source=StructureSource.MARKDOWN,
                ),
                metadata={"links": self._extract_links(token)},
            )

        for image in images:
            self._create_image_block(image)

    def _parse_list_item_content(
        self,
        token: Token,
    ) -> None:

        list_context = self._list_stack[-1]

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

        parent_id = self._resolve_list_parent()

        block = self._create_block(
            block_type=BlockType.LIST_ITEM,
            text=text,
            structure=StructureInfo(
                parent_id=parent_id,
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
            list_info=ListInfo(
                type=list_context["type"],
                level=len(self._list_stack),
                index=item_index,
                marker=marker,
            ),
            metadata={"links": self._extract_links(token)},
        )

        self._current_list_item_id = block.id

        list_context["last_item_id"] = block.id

    def _open_list(
        self,
        list_type: ListType,
        start: int = 1,
    ) -> None:

        parent_item_id = None

        if self._list_stack:
            parent_item_id = self._list_stack[-1].get("last_item_id")

        self._list_stack.append(
            {
                "type": list_type,
                "start": start,
                "current_index": 0,
                "last_item_id": None,
                "parent_item_id": parent_item_id,
            }
        )

    def _close_list(self) -> None:

        if self._list_stack:
            self._list_stack.pop()

    def _resolve_list_parent(
        self,
    ) -> str | None:

        current = self._list_stack[-1]

        parent_item_id = current.get("parent_item_id")

        if parent_item_id:
            return parent_item_id

        return self._current_heading_id()

    def _parse_code(
        self,
        token: Token,
    ) -> None:

        language = token.info.strip() or None

        self._create_block(
            block_type=BlockType.CODE,
            text=token.content.rstrip(),
            structure=StructureInfo(
                parent_id=self._current_heading_id(),
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
            metadata={"language": language},
        )

    def _parse_table(
        self,
        tokens: list[Token],
        index: int,
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
            block_type=BlockType.TABLE,
            text=text,
            structure=StructureInfo(
                parent_id=self._current_heading_id(),
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
            table_info=TableInfo(rows=rows),
        )

        return index

    def _resolve_heading_parent(
        self,
        level: int,
    ) -> str | None:

        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()

        if not self._heading_stack:
            return None

        return self._heading_stack[-1][1]

    def _current_heading_id(
        self,
    ) -> str | None:

        if not self._heading_stack:
            return None

        return self._heading_stack[-1][1]

    def _create_image_block(
        self,
        image: dict,
    ) -> None:

        self._create_block(
            block_type=BlockType.IMAGE,
            text=image["alt"],
            structure=StructureInfo(
                parent_id=self._current_heading_id(),
                confidence=1.0,
                source=StructureSource.MARKDOWN,
            ),
            image_info=ImageInfo(
                path=image["path"],
                caption=image["alt"] or None,
            ),
        )

    def _create_block(
        self,
        block_type: BlockType,
        text: str,
        structure: StructureInfo,
        list_info: ListInfo | None = None,
        table_info: TableInfo | None = None,
        image_info: ImageInfo | None = None,
        metadata: dict | None = None,
    ) -> Block:

        self._order += 1

        block = Block(
            id=f"block_{uuid4().hex}",
            type=block_type,
            text=text,
            order=self._order,
            bbox=None,
            structure=structure,
            list=list_info,
            table=table_info,
            image=image_info,
            metadata=metadata or {},
        )

        self._blocks.append(block)

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
            if child.type == "text" or child.type == "code_inline":
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
