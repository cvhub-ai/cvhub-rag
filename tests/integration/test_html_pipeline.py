from pathlib import Path

from src.assets.manager import AssetManager
from src.assets.resolver import AssetResolver
from src.parsers.html.parser import HtmlParser

TEST_DOCUMENT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "documents"
    / "test_computer_vision_html.html"
)


def test_html_parser_pipeline(tmp_path: Path) -> None:
    asset_resolver = AssetResolver()

    asset_manager = AssetManager(
        root_dir=tmp_path / "assets",
    )

    parser = HtmlParser(
        asset_resolver=asset_resolver,
        asset_manager=asset_manager,
    )

    document = parser.parse(TEST_DOCUMENT)

    assert document.document_id == TEST_DOCUMENT.stem
    assert document.pages
    assert document.pages[0].blocks
