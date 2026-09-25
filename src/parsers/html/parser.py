from pathlib import Path
from uuid import uuid4

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from src.models.enums import (
    BlockType,
    FileType,
    ListType,
    ParserType,
    StructureSource,
)
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
from src.parsers.html.state import (
    HtmlListState,
    HtmlParseState,
)


class HtmlParser(BaseDocumentParser):
    def parse(
        self,
        file_path: str | Path,
    ) -> ParsedDocument:
        path = Path(file_path)

        content = path.read_text(encoding="utf-8")

        soup = BeautifulSoup(content, "lxml")

        parser_state = HtmlParseState()

        root = soup.body or soup

        self._parse_node(root, parser_state)

        page = Page(page_number=1, width=None, height=None, blocks=parser_state.blocks)

        return ParsedDocument(
            document_id=path.stem,
            source=SourceInfo(
                file_name=path.name, file_type=FileType.HTML, parser=ParserType.HTML
            ),
            pages=[page],
        )

    def _parse_node(
        self,
        node: Tag,
        parser_state: HtmlParseState,
    ) -> None:
        if self._is_heading(node):
            self._parse_heading(node, parser_state)
            return

        if node.name == "p":
            self._parse_paragraph(node, parser_state)
            return

        if node.name == "ul":
            self._parse_list(node, parser_state, ListType.UNORDERED)
            return

        if node.name == "ol":
            self._parse_list(node, parser_state, ListType.ORDERED)
            return

        if node.name == "table":
            self._parse_table(node, parser_state)
            return

        if node.name == "img":
            self._parse_image(node, parser_state)
            return

        for child in node.children:
            if isinstance(child, Tag):
                self._parse_node(child, parser_state)
            elif isinstance(child, NavigableString):
                self._parse_text(child, parser_state)

    @staticmethod
    def _is_heading(
        node: Tag,
    ) -> bool:
        return node.name in {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }

    def _parse_heading(
        self,
        node: Tag,
        parser_state: HtmlParseState,
    ) -> None:
        level = int(node.name.removeprefix("h"))

        text = node.get_text(separator=" ", strip=True)

        parent_id = self._resolve_heading_parent(parser_state, level)

        block = self._create_block(
            parser_state=parser_state,
            block_type=BlockType.HEADING,
            text=text,
            structure=StructureInfo(
                parent_id=parent_id,
                level=level,
                confidence=1.0,
                source=StructureSource.PARSER,
            ),
        )

        parser_state.heading_stack.append((level, block.id))

    def _parse_paragraph(
        self,
        node: Tag,
        parser_state: HtmlParseState,
    ) -> None:
        text = node.get_text(separator=" ", strip=True)

        if not text:
            return

        self._create_block(
            parser_state=parser_state,
            block_type=BlockType.PARAGRAPH,
            text=text,
            structure=StructureInfo(
                parent_id=self._current_heading_id(parser_state),
                confidence=1.0,
                source=StructureSource.PARSER,
            ),
        )

    def _parse_text(
        self,
        node: NavigableString,
        parser_state: HtmlParseState,
    ) -> None:
        text = str(node).strip()

        if not text:
            return

        self._create_block(
            parser_state=parser_state,
            block_type=BlockType.PARAGRAPH,
            text=text,
            structure=StructureInfo(
                parent_id=self._current_heading_id(parser_state),
                confidence=1.0,
                source=StructureSource.PARSER,
            ),
        )

    def _parse_list(
        self,
        node: Tag,
        parser_state: HtmlParseState,
        list_type: ListType,
    ) -> None:
        start = 1

        if list_type == ListType.ORDERED:
            start_value = node.get("start")

            if start_value is not None:
                start = int(str(start_value))

        parent_item_id = None

        if parser_state.list_stack:
            parent_item_id = parser_state.list_stack[-1].last_item_id

        parser_state.list_stack.append(
            HtmlListState(
                type=list_type,
                start=start,
                parent_item_id=parent_item_id,
            )
        )

        for child in node.children:
            if isinstance(child, Tag) and child.name == "li":
                self._parse_list_item(
                    child,
                    parser_state,
                )

        parser_state.list_stack.pop()

    def _parse_list_item(
        self,
        node: Tag,
        parser_state: HtmlParseState,
    ) -> None:
        list_context = parser_state.list_stack[-1]

        text_parts: list[str] = []

        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()

                if text:
                    text_parts.append(text)

            elif isinstance(child, Tag):
                if child.name not in {"ul", "ol"}:
                    text = child.get_text(separator=" ", strip=True)

                    if text:
                        text_parts.append(text)

        text = " ".join(text_parts).strip()

        list_context.current_index += 1

        if list_context.type == ListType.ORDERED:
            item_index = list_context.start + list_context.current_index - 1

            marker = f"{item_index}."
        else:
            item_index = None
            marker = "-"

        parent_id = list_context.parent_item_id or self._current_heading_id(
            parser_state
        )

        block = self._create_block(
            parser_state=parser_state,
            block_type=BlockType.LIST_ITEM,
            text=text,
            structure=StructureInfo(
                parent_id=parent_id,
                confidence=1.0,
                source=StructureSource.PARSER,
            ),
            list_info=ListInfo(
                type=list_context.type,
                level=len(parser_state.list_stack),
                index=item_index,
                marker=marker,
            ),
        )

        list_context.last_item_id = block.id

        for child in node.children:
            if not isinstance(child, Tag):
                continue

            if child.name == "ul":
                self._parse_list(
                    child,
                    parser_state,
                    ListType.UNORDERED,
                )

            elif child.name == "ol":
                self._parse_list(
                    child,
                    parser_state,
                    ListType.ORDERED,
                )

    def _parse_table(
        self,
        node: Tag,
        parser_state: HtmlParseState,
    ) -> None:
        rows: list[list[str]] = []

        for row in node.find_all("tr"):
            cells: list[str] = []

            for cell in row.find_all(
                ["th", "td"],
                recursive=False,
            ):
                text = cell.get_text(separator=" ", strip=True)

                cells.append(text)

            if cells:
                rows.append(cells)

        if not rows:
            return

        text = "\n".join(" | ".join(row) for row in rows)

        self._create_block(
            parser_state=parser_state,
            block_type=BlockType.TABLE,
            text=text,
            structure=StructureInfo(
                parent_id=self._current_heading_id(parser_state),
                confidence=1.0,
                source=StructureSource.PARSER,
            ),
            table_info=TableInfo(
                rows=rows,
            ),
        )

    def _parse_image(
        self,
        node: Tag,
        parser_state: HtmlParseState,
    ) -> None:
        src = node.get("src")
        alt = node.get("alt")

        path = src if isinstance(src, str) else ""
        caption = alt if isinstance(alt, str) else None

        self._create_block(
            parser_state=parser_state,
            block_type=BlockType.IMAGE,
            text=caption or "",
            structure=StructureInfo(
                parent_id=self._current_heading_id(parser_state),
                confidence=1.0,
                source=StructureSource.PARSER,
            ),
            image_info=ImageInfo(
                path=path,
                caption=caption,
            ),
        )

    @staticmethod
    def _resolve_heading_parent(
        parser_state: HtmlParseState,
        level: int,
    ) -> str | None:
        while parser_state.heading_stack and parser_state.heading_stack[-1][0] >= level:
            parser_state.heading_stack.pop()

        if not parser_state.heading_stack:
            return None

        return parser_state.heading_stack[-1][1]

    @staticmethod
    def _current_heading_id(
        parser_state: HtmlParseState,
    ) -> str | None:
        if not parser_state.heading_stack:
            return None

        return parser_state.heading_stack[-1][1]

    @staticmethod
    def _create_block(
        parser_state: HtmlParseState,
        block_type: BlockType,
        text: str,
        structure: StructureInfo,
        list_info: ListInfo | None = None,
        table_info: TableInfo | None = None,
        image_info: ImageInfo | None = None,
    ) -> Block:
        block = Block(
            id=f"block_{uuid4().hex}",
            type=block_type,
            text=text,
            order=parser_state.next_order(),
            bbox=None,
            structure=structure,
            list=list_info,
            table=table_info,
            image=image_info,
        )

        parser_state.blocks.append(block)

        return block
