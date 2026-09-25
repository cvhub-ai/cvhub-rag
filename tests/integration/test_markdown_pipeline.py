import json
from pathlib import Path

from src.assets.manager import AssetManager
from src.assets.resolver import AssetResolver
from src.parsers.markdown.parser import MarkdownParser


def test_markdown_parser_pipeline() -> None:
    input_path = Path("./input/markdown/Feature Matching Zusammenfassung.md")

    output_path = Path("./output/test_markdown_output.json")

    asset_resolver = AssetResolver()

    asset_manager = AssetManager(
        root_dir="./output/assets",
    )

    parser = MarkdownParser(
        asset_resolver=asset_resolver,
        asset_manager=asset_manager,
    )

    document = parser.parse(input_path)

    assert document.document_id == input_path.stem
    assert document.pages
    assert document.pages[0].blocks

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            document.to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
