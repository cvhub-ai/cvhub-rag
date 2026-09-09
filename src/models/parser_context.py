from dataclasses import dataclass, field

from src.models.parser_document import Block


@dataclass
class ParserContext:
    blocks: list[Block] = field(default_factory=list)
    order: int = 0

    heading_stack: list[tuple[int, str]] = field(default_factory=list)

    list_stack: list[dict] = field(default_factory=list)
    current_list_item_id: str | None = None

    def next_order(self) -> int:
        self.order += 1
        return self.order
