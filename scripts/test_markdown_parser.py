import json
from pathlib import Path

from src.parsers.markdown_parser import MarkdownParser


def main() -> None:
    input_path = Path("./input/Feature Matching Zusammenfassung.md")
    output_path = Path("./output/output.json")

    parser = MarkdownParser()

    document = parser.parse(input_path)

    result = document.to_dict()

    json_text = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )

    output_path.write_text(
        json_text,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
