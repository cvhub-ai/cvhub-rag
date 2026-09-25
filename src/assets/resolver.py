from pathlib import Path
from urllib.parse import urlparse


class AssetResolver:
    def resolve(
        self,
        reference: str,
        source_file: str | Path,
    ) -> Path | str:
        if self._is_remote(reference):
            return reference

        reference_path = Path(reference)

        if reference_path.is_absolute():
            return reference_path

        source_path = Path(source_file)

        return (source_path.parent / reference_path).resolve()

    @staticmethod
    def _is_remote(
        reference: str,
    ) -> bool:
        parsed = urlparse(reference)

        return parsed.scheme in {
            "http",
            "https",
        }
