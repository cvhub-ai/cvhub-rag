from abc import ABC, abstractmethod
from pathlib import Path

from models.parsed_document import ParsedDocument


class BaseDocumentParser(ABC):
    @abstractmethod
    def parse(
        self,
        file_path: str | Path,
    ) -> ParsedDocument:
        pass
