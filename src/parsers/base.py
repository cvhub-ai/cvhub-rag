from abc import ABC, abstractmethod
from pathlib import Path

from src.models.parser_document import ParsedDocument


class BaseDocumentParser(ABC):
    @abstractmethod
    def parse(
        self,
        file_path: str | Path,
    ) -> ParsedDocument:
        pass
