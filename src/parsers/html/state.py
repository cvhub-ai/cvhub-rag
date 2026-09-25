from dataclasses import dataclass, field

from src.models.enums import ListType
from src.models.parser_document import Block


@dataclass
class HtmlListState:
    type: ListType
    start: int = 1
    current_index: int = 0
    last_item_id: str | None = None
    parent_item_id: str | None = None


@dataclass
class HtmlParseState:
    blocks: list[Block] = field(default_factory=list)

    order: int = 0

    heading_stack: list[tuple[int, str]] = field(default_factory=list)

    list_stack: list[HtmlListState] = field(default_factory=list)

    def next_order(self) -> int:
        self.order += 1
        return self.order
