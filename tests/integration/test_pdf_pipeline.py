from pathlib import Path

from src.assets.manager import AssetManager
from src.parsers.pdf.parser import PDFParser

TEST_DOCUMENT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "documents"
    / "test_computer_vision_pdf.pdf"
)


def test_pdf_parser_pipeline(tmp_path: Path) -> None:
    asset_manager = AssetManager(
        root_dir=tmp_path / "assets",
    )

    parser = PDFParser(
        asset_manager=asset_manager,
    )

    document = parser.parse(TEST_DOCUMENT)

    assert document.document_id == TEST_DOCUMENT.stem
    assert document.pages
    assert any(page.blocks for page in document.pages)
