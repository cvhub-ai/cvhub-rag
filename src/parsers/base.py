from abc import ABC, abstractmethod
from pathlib import Path

from src.models.document import UnifiedDocument


class BaseDocumentParser(ABC):
    @abstractmethod
    def parse(
        self,
        file_path: str | Path,
    ) -> UnifiedDocument:
        pass
