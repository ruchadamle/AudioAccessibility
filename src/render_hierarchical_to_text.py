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
    text = text.rstrip(".")
    return text


def join_phrases(items: list[str]) -> str:
    items = [clean_sentence(x) for x in items if str(x).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def extract_chart_fields(data: dict) -> dict:
    chart_type = data.get("chart_type", "").strip()
    title = data.get("title", "").strip()
    axes = data.get("axes", {}) or {}
    return {
        "chart_type": chart_type,
        "title": title,
        "x_axis": axes.get("x_axis_title", "").strip(),
        "y_axis": axes.get("y_axis_title", "").strip(),
        "unit": axes.get("y_axis_unit", "").strip(),
        "key_values": data.get("key_values", []) or [],
        "comparisons": data.get("comparisons", []) or [],
        "trends": data.get("trends", []) or [],
        "takeaway": data.get("takeaway", "").strip(),
    }


def build_axis_bits(x_axis: str, y_axis: str, unit: str, *, style: str) -> list[str]:
    axis_bits = []
    if x_axis:
        if style == "concise":
            axis_bits.append(f"the x-axis represents {x_axis}")
        else:
            axis_bits.append(f"the x-axis shows {x_axis}")
    if y_axis and unit:
        if style == "concise":
            axis_bits.append(f"the y-axis represents {y_axis} measured in {unit}")
        else:
            axis_bits.append(f"the y-axis shows {y_axis} in {unit}")
    elif y_axis:
        if style == "concise":
            axis_bits.append(f"the y-axis represents {y_axis}")
        else:
            axis_bits.append(f"the y-axis shows {y_axis}")
    return axis_bits


def render_concise(data: dict) -> str:
    fields = extract_chart_fields(data)
    chart_type = fields["chart_type"]
    title = fields["title"]
    x_axis = fields["x_axis"]
    y_axis = fields["y_axis"]
    unit = fields["unit"]
    key_values = fields["key_values"]
    comparisons = fields["comparisons"]
    trends = fields["trends"]
    takeaway = fields["takeaway"]

    parts = []

    intro_bits = []
    if chart_type:
        intro_bits.append(f"This {chart_type}")
    else:
        intro_bits.append("This chart")

    if title:
        intro_bits.append(f'shows "{title}"')
    else:
        intro_bits.append("shows the displayed data")

    parts.append(" ".join(intro_bits) + ".")

    axis_bits = build_axis_bits(x_axis, y_axis, unit, style="concise")
    if axis_bits:
        parts.append(" ".join(["In this chart,", join_phrases(axis_bits) + "."]))

    if key_values:
        parts.append("Key values include " + join_phrases(key_values) + ".")

    if comparisons:
        parts.append("Notable comparisons include " + join_phrases(comparisons) + ".")

    if trends:
        parts.append("The overall pattern is that " + join_phrases(trends) + ".")

    if takeaway:
        parts.append("Overall, " + clean_sentence(takeaway) + ".")

    return " ".join(parts)


def render_detailed(data: dict) -> str:
    fields = extract_chart_fields(data)
    chart_type = fields["chart_type"]
    title = fields["title"]
    x_axis = fields["x_axis"]
    y_axis = fields["y_axis"]
    unit = fields["unit"]
    key_values = fields["key_values"]
    comparisons = fields["comparisons"]
    trends = fields["trends"]
    takeaway = fields["takeaway"]

    sentences = []

    if chart_type and title:
        sentences.append(f'This {chart_type} is titled "{title}".')
    elif chart_type:
        sentences.append(f"This is a {chart_type}.")
    elif title:
        sentences.append(f'The chart is titled "{title}".')
    else:
        sentences.append("This chart presents the displayed data.")

    axis_parts = build_axis_bits(x_axis, y_axis, unit, style="detailed")
    if axis_parts:
        sentences.append("The chart is organized so that " + join_phrases(axis_parts) + ".")

    if key_values:
        sentences.append("Important values include " + join_phrases(key_values) + ".")

    if comparisons:
        sentences.append("The chart also shows that " + join_phrases(comparisons) + ".")

    if trends:
        sentences.append("A visible overall pattern is that " + join_phrases(trends) + ".")

    if takeaway:
        sentences.append("The main takeaway is that " + clean_sentence(takeaway) + ".")

    return " ".join(sentences)


def render_file(input_path: Path, concise_out: Path, detailed_out: Path) -> None:
    data = load_json(input_path)

    if "error" in data:
        concise_text = f"Could not render this file because the JSON output was invalid. Raw output: {data.get('raw_output', '')}"
        detailed_text = concise_text
    else:
        concise_text = render_concise(data)
        detailed_text = render_detailed(data)

    concise_out.write_text(concise_text, encoding="utf-8")
    detailed_out.write_text(detailed_text, encoding="utf-8")


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
    concise_dir = output_dir / "concise"
    detailed_dir = output_dir / "detailed"

    concise_dir.mkdir(parents=True, exist_ok=True)
    detailed_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    for json_file in json_files:
        concise_out = concise_dir / f"{json_file.stem}.txt"
        detailed_out = detailed_dir / f"{json_file.stem}.txt"
        render_file(json_file, concise_out, detailed_out)

    print(f"Rendered {len(json_files)} files to {output_dir}")


if __name__ == "__main__":
    main()
