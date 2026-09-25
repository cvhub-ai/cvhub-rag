from pathlib import Path
from shutil import copy2
from uuid import uuid4

from PIL.Image import Image


class AssetManager:
    def __init__(
        self,
        root_dir: str | Path,
    ) -> None:
        self._root_dir = Path(root_dir)

    def save_file(
        self,
        source_path: str | Path,
        document_id: str,
    ) -> str:
        source = Path(source_path)

        suffix = source.suffix

        file_name = f"{uuid4().hex}{suffix}"

        relative_path = Path(document_id) / "images" / file_name

        target_path = self._root_dir / relative_path

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        copy2(
            source,
            target_path,
        )

        return relative_path.as_posix()

    def save_image(
        self,
        image: Image,
        document_id: str,
        image_format: str = "PNG",
    ) -> str:
        suffix = self._get_suffix(image_format)

        file_name = f"{uuid4().hex}{suffix}"

        relative_path = Path(document_id) / "images" / file_name

        target_path = self._root_dir / relative_path

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            target_path,
            format=image_format,
        )

        return relative_path.as_posix()

    @staticmethod
    def _get_suffix(
        image_format: str,
    ) -> str:
        match image_format.upper():
            case "PNG":
                return ".png"

            case "JPEG" | "JPG":
                return ".jpg"

            case "WEBP":
                return ".webp"

            case _:
                raise ValueError(f"Unsupported image format: {image_format}")
