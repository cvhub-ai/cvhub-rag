import json
from pathlib import Path

from src.parsers.pdf.parser import PDFParser


def main() -> None:
    input_path = Path("./input/pdf/mit_deep learning for computer vision.pdf")
    output_path = Path("./output/test_pdf_output.json")

    parser = PDFParser()

    document = parser.parse(input_path)

    result = document.to_dict()

    json_text = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json_text,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
