import argparse
import json
import re
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_sentence(text: str) -> str:
    text = clean_text(text)
    text = text.rstrip(". ")
    return text


def ensure_period(text: str) -> str:
    text = clean_sentence(text)
    if not text:
        return ""
    return text + "."


def sentence_key(text: str) -> str:
    text = clean_sentence(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_sentences(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        item = clean_sentence(item)
        if not item:
            continue
        key = sentence_key(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def soften_after_prefix(text: str) -> str:
    text = clean_sentence(text)
    if not text:
        return ""

    first_word, separator, rest = text.partition(" ")
    common_starters = {
        "a",
        "an",
        "the",
        "this",
        "these",
        "those",
        "it",
        "its",
        "there",
        "values",
        "most",
    }

    if first_word.lower() in common_starters:
        return first_word.lower() + (separator + rest if separator else "")
    return text


def join_clauses(items: list[str]) -> str:
    items = [clean_sentence(item) for item in items if clean_sentence(item)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    first_item = items[0]
    remaining_items = [soften_after_prefix(item) for item in items[1:]]
    return "; ".join([first_item] + remaining_items)


def build_group_sentence(prefix: str, items: list[str]) -> str:
    content = join_clauses(items)
    if not content:
        return ""
    return ensure_period(f"{prefix} {soften_after_prefix(content)}")


def normalize_chart_type(chart_type: str) -> str:
    chart_type = clean_sentence(chart_type).replace("_", " ")
    chart_type = re.sub(r"\s+", " ", chart_type)
    return chart_type


def normalize_title(title: str) -> str:
    return clean_sentence(title)


def build_intro(chart_type: str, title: str, axes: dict) -> str:
    chart_type = normalize_chart_type(chart_type)
    title = normalize_title(title)

    x_axis = clean_sentence(axes.get("x_axis_title", ""))
    y_axis = clean_sentence(axes.get("y_axis_title", ""))
    unit = clean_sentence(axes.get("y_axis_unit", ""))

    if chart_type and title:
        intro = f'This {chart_type} is titled "{title}"'
    elif chart_type:
        intro = f"This {chart_type} shows the data"
    elif title:
        intro = f'This chart is titled "{title}"'
    else:
        intro = "This chart shows the data"

    if x_axis and y_axis and unit:
        intro += f", with {x_axis} on the x-axis and {y_axis} on the y-axis in {unit}"
    elif x_axis and y_axis:
        intro += f", with {x_axis} on the x-axis and {y_axis} on the y-axis"
    elif x_axis:
        intro += f", with {x_axis} on the x-axis"
    elif y_axis and unit:
        intro += f", with {y_axis} on the y-axis in {unit}"
    elif y_axis:
        intro += f", with {y_axis} on the y-axis"

    return ensure_period(intro)


def render_natural(data: dict) -> str:
    chart_type = data.get("chart_type", "")
    title = data.get("title", "")
    axes = data.get("axes", {}) or {}
    key_values = dedupe_sentences(data.get("key_values", []) or [])
    comparisons = dedupe_sentences(data.get("comparisons", []) or [])
    trends = dedupe_sentences(data.get("trends", []) or [])
    takeaway = clean_sentence(data.get("takeaway", ""))

    sentences = []

    intro = build_intro(chart_type, title, axes)
    if intro:
        sentences.append(intro)

    used_keys = {sentence_key(s) for s in sentences}

    selected_key_values = []
    for item in key_values[:3]:
        key = sentence_key(item)
        if key and key not in used_keys:
            selected_key_values.append(item)
            used_keys.add(key)

    selected_comparisons = []
    for item in comparisons[:2]:
        key = sentence_key(item)
        if key and key not in used_keys:
            selected_comparisons.append(item)
            used_keys.add(key)

    selected_trends = []
    for item in trends[:1]:
        key = sentence_key(item)
        if key and key not in used_keys:
            selected_trends.append(item)
            used_keys.add(key)

    key_values_sentence = build_group_sentence("Notably,", selected_key_values)
    if key_values_sentence:
        sentences.append(key_values_sentence)

    comparisons_sentence = build_group_sentence("In comparison,", selected_comparisons)
    if comparisons_sentence:
        sentences.append(comparisons_sentence)

    trends_sentence = build_group_sentence("Overall,", selected_trends)
    if trends_sentence:
        sentences.append(trends_sentence)

    if takeaway:
        sentences.append(ensure_period(takeaway))

    return " ".join(sentences)


def render_file(input_path: Path, output_path: Path) -> None:
    data = load_json(input_path)

    if "error" in data:
        text = (
            "Could not render this file because the JSON output was invalid. "
            f"Raw output: {data.get('raw_output', '')}"
        )
    else:
        text = render_natural(data)

    output_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
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
