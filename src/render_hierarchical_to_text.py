import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_sentence(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    text = text.rstrip(". ")
    return text


def lower_first_if_needed(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[0].isupper():
        return text[0].lower() + text[1:]
    return text


def merge_sentences(parts: list[str]) -> str:
    cleaned = []
    for part in parts:
        part = clean_sentence(part)
        if part:
            cleaned.append(part + ".")
    return " ".join(cleaned)


def render_natural(data: dict) -> str:
    chart_type = clean_sentence(data.get("chart_type", ""))
    title = clean_sentence(data.get("title", ""))
    key_values = data.get("key_values", []) or []
    comparisons = data.get("comparisons", []) or []
    trends = data.get("trends", []) or []
    takeaway = clean_sentence(data.get("takeaway", ""))

    parts = []

    if chart_type and title:
        parts.append(f'This {chart_type} shows "{lower_first_if_needed(title)}"')
    elif chart_type:
        parts.append(f"This {chart_type} presents the data")
    elif title:
        parts.append(f'This chart shows "{lower_first_if_needed(title)}"')
    else:
        parts.append("This chart presents the data")

    body_parts = []

    if key_values:
        body_parts.extend([clean_sentence(x) for x in key_values if clean_sentence(x)])

    if comparisons:
        body_parts.extend([clean_sentence(x) for x in comparisons if clean_sentence(x)])

    if trends:
        body_parts.extend([clean_sentence(x) for x in trends if clean_sentence(x)])

    if body_parts:
        parts.append(" ".join(body_parts))

    if takeaway:
        parts.append(takeaway)

    return merge_sentences(parts)


def render_file(input_path: Path, output_path: Path) -> None:
    data = load_json(input_path)

    if "error" in data:
        text = f"Could not render this file because the JSON output was invalid. Raw output: {data.get('raw_output', '')}"
    else:
        text = render_natural(data)

    output_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing hierarchical JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save rendered text files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    for json_file in json_files:
        output_path = output_dir / f"{json_file.stem}.txt"
        render_file(json_file, output_path)

    print(f"Rendered {len(json_files)} files to {output_dir}")


if __name__ == "__main__":
    main()