from pathlib import Path
from uuid import uuid4

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from src.assets.manager import AssetManager
from src.assets.resolver import AssetResolver
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

IGNORE_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "nav",
}


class HtmlParser(BaseDocumentParser):
    def __init__(
        self,
        asset_resolver: AssetResolver,
        asset_manager: AssetManager,
    ) -> None:
        self._asset_resolver = asset_resolver
        self._asset_manager = asset_manager

    def parse(
        self,
        file_path: str | Path,
    ) -> ParsedDocument:
        path = Path(file_path)

        content = path.read_text(
            encoding="utf-8",
        )

        soup = BeautifulSoup(
            content,
            "lxml",
        )

        parser_state = HtmlParseState()

        root = soup.body or soup

        self._parse_node(
            root,
            parser_state,
            source_file=path,
        )

        page = Page(
            page_number=1,
            width=None,
            height=None,
            blocks=parser_state.blocks,
        )

        return ParsedDocument(
            document_id=path.stem,
            source=SourceInfo(
                file_name=path.name,
                file_type=FileType.HTML,
                parser=ParserType.HTML,
            ),
            pages=[page],
        )

    def _parse_node(
        self,
        node: Tag,
        parser_state: HtmlParseState,
        source_file: Path,
    ) -> None:
        if node.name in IGNORE_TAGS:
            return

        if self._is_heading(node):
            self._parse_heading(
                node,
                parser_state,
            )
            return

        if node.name == "p":
            self._parse_paragraph(
                node,
                parser_state,
                source_file,
            )
            return

        if node.name == "ul":
            self._parse_list(
                node,
                parser_state,
                ListType.UNORDERED,
                source_file,
            )
            return

        if node.name == "ol":
            self._parse_list(
                node,
                parser_state,
                ListType.ORDERED,
                source_file,
            )
            return

        if node.name == "table":
            self._parse_table(
                node,
                parser_state,
            )
            return

        if node.name == "img":
            self._parse_image(
                node=node,
                parser_state=parser_state,
                source_file=source_file,
                parent_id=self._current_heading_id(parser_state),
            )
            return

        if node.name == "pre":
            self._parse_code(
                node,
                parser_state,
            )
            return

        for child in node.children:
            if isinstance(child, Tag):
                self._parse_node(
                    child,
                    parser_state,
                    source_file,
                )

            elif isinstance(child, NavigableString):
                self._parse_text(
                    child,
                    parser_state,
                )

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

        text = node.get_text(
            separator=" ",
            strip=True,
        )

        parent_id = self._resolve_heading_parent(
            parser_state,
            level,
        )

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

        parser_state.heading_stack.append(
            (
                level,
                block.id,
            )
        )

    def _parse_paragraph(
        self,
        node: Tag,
        parser_state: HtmlParseState,
        source_file: Path,
    ) -> None:
        text = node.get_text(
            separator=" ",
            strip=True,
        )

        parent_id = self._current_heading_id(parser_state)

        if text:
            self._create_block(
                parser_state=parser_state,
                block_type=BlockType.PARAGRAPH,
                text=text,
                structure=StructureInfo(
                    parent_id=parent_id,
                    confidence=1.0,
                    source=StructureSource.PARSER,
                ),
                metadata={
                    "links": self._extract_links(node),
                },
            )

        for image in node.find_all("img"):
            self._parse_image(
                node=image,
                parser_state=parser_state,
                source_file=source_file,
                parent_id=parent_id,
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
        source_file: Path,
    ) -> None:
        start = 1

        if list_type == ListType.ORDERED:
            start_value = node.get("start")

            if isinstance(start_value, str):
                start = int(start_value)

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
                    source_file,
                )

        parser_state.list_stack.pop()

    def _parse_list_item(
        self,
        node: Tag,
        parser_state: HtmlParseState,
        source_file: Path,
    ) -> None:
        list_context = parser_state.list_stack[-1]

        text_parts: list[str] = []
        images: list[Tag] = []

        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()

                if text:
                    text_parts.append(text)

            elif isinstance(child, Tag):
                if child.name in {
                    "ul",
                    "ol",
                }:
                    continue

                if child.name == "img":
                    images.append(child)
                    continue

                text = child.get_text(
                    separator=" ",
                    strip=True,
                )

                if text:
                    text_parts.append(text)

                images.extend(child.find_all("img"))

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
            metadata={
                "links": self._extract_list_item_links(node),
            },
        )

        list_context.last_item_id = block.id

        for image in images:
            self._parse_image(
                node=image,
                parser_state=parser_state,
                source_file=source_file,
                parent_id=block.id,
            )

        for child in node.children:
            if not isinstance(child, Tag):
                continue

            if child.name == "ul":
                self._parse_list(
                    child,
                    parser_state,
                    ListType.UNORDERED,
                    source_file,
                )

            elif child.name == "ol":
                self._parse_list(
                    child,
                    parser_state,
                    ListType.ORDERED,
                    source_file,
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
                text = cell.get_text(
                    separator=" ",
                    strip=True,
                )

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
        source_file: Path,
        parent_id: str | None,
    ) -> None:
        src = node.get("src")
        alt = node.get("alt")

        if not isinstance(src, str) or not src:
            return

        caption = alt if isinstance(alt, str) else None

        resolved = self._asset_resolver.resolve(
            reference=src,
            source_file=source_file,
        )

        if isinstance(resolved, Path):
            managed_path = self._asset_manager.save_file(
                source_path=resolved,
                document_id=source_file.stem,
            )
        else:
            managed_path = resolved

        self._create_block(
            parser_state=parser_state,
            block_type=BlockType.IMAGE,
            text=caption or "",
            structure=StructureInfo(
                parent_id=parent_id,
                confidence=1.0,
                source=StructureSource.PARSER,
            ),
            image_info=ImageInfo(
                path=managed_path,
                caption=caption,
            ),
        )

    def _parse_code(
        self,
        node: Tag,
        parser_state: HtmlParseState,
    ) -> None:
        code_node = node.find("code")

        if isinstance(code_node, Tag):
            text = code_node.get_text().rstrip()
            language = self._extract_code_language(code_node)
        else:
            text = node.get_text().rstrip()
            language = None

        if not text:
            return

        self._create_block(
            parser_state=parser_state,
            block_type=BlockType.CODE,
            text=text,
            structure=StructureInfo(
                parent_id=self._current_heading_id(parser_state),
                confidence=1.0,
                source=StructureSource.PARSER,
            ),
            metadata={
                "language": language,
            },
        )

    @staticmethod
    def _extract_code_language(
        node: Tag,
    ) -> str | None:
        classes = node.get("class")

        if not isinstance(classes, list):
            return None

        for class_name in classes:
            if not isinstance(class_name, str):
                continue

            if class_name.startswith("language-"):
                return class_name.removeprefix("language-")

        return None

    @staticmethod
    def _extract_links(
        node: Tag,
    ) -> list[dict]:
        links: list[dict] = []

        for link in node.find_all("a"):
            href = link.get("href")

            if not isinstance(href, str):
                continue

            links.append(
                {
                    "url": href,
                    "text": link.get_text(
                        separator=" ",
                        strip=True,
                    ),
                }
            )

        return links

    def _extract_list_item_links(
        self,
        node: Tag,
    ) -> list[dict]:
        links: list[dict] = []

        for child in node.children:
            if not isinstance(child, Tag):
                continue

            if child.name in {
                "ul",
                "ol",
            }:
                continue

            if child.name == "a":
                href = child.get("href")

                if isinstance(href, str):
                    links.append(
                        {
                            "url": href,
                            "text": child.get_text(
                                separator=" ",
                                strip=True,
                            ),
                        }
                    )

                continue

            links.extend(self._extract_links(child))

        return links

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
        metadata: dict | None = None,
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
            metadata=metadata or {},
        )

        parser_state.blocks.append(block)

        return block
