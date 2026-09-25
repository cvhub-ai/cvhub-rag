import json
from pathlib import Path

from src.assets.manager import AssetManager
from src.parsers.pdf.parser import PDFParser


def test_pdf_parser_pipeline() -> None:
    input_path = Path("./input/pdf/mit_deep learning for computer vision.pdf")

    output_path = Path("./output/test_pdf_output.json")

    asset_manager = AssetManager(
        root_dir="./output/assets",
    )

    parser = PDFParser(
        asset_manager=asset_manager,
    )

    document = parser.parse(input_path)

    assert document.document_id == input_path.stem
    assert document.pages
    assert any(page.blocks for page in document.pages)

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
