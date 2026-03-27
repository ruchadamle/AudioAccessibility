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
    return clean_text(text).rstrip(". ")


def ensure_period(text: str) -> str:
    text = clean_sentence(text)
    return f"{text}." if text else ""


def sentence_key(text: str) -> str:
    text = clean_sentence(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_for_overlap(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", clean_sentence(text).lower())
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "shows",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    return {word for word in words if word not in stopwords}


def is_redundant(candidate: str, existing: list[str]) -> bool:
    candidate_key = sentence_key(candidate)
    candidate_tokens = tokenize_for_overlap(candidate)

    for current in existing:
        current_key = sentence_key(current)
        if not current_key:
            continue
        if candidate_key == current_key:
            return True
        if candidate_key and (candidate_key in current_key or current_key in candidate_key):
            return True

        current_tokens = tokenize_for_overlap(current)
        if candidate_tokens and current_tokens:
            overlap = len(candidate_tokens & current_tokens)
            denominator = min(len(candidate_tokens), len(current_tokens))
            if denominator and overlap / denominator >= 0.8:
                return True

    return False


def extract_rank_claim(text: str) -> tuple[str, str] | None:
    text = clean_sentence(text)
    match = re.search(
        r"^(?P<entity>.+?) has the (?P<rank>highest|lowest|second highest|third highest) ",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    entity = clean_sentence(match.group("entity")).lower()
    rank = clean_sentence(match.group("rank")).lower()
    return entity, rank


def contradicts_existing(candidate: str, existing: list[str]) -> bool:
    candidate_claim = extract_rank_claim(candidate)
    if not candidate_claim:
        return False

    candidate_entity, candidate_rank = candidate_claim
    for current in existing:
        current_claim = extract_rank_claim(current)
        if not current_claim:
            continue
        current_entity, current_rank = current_claim
        if candidate_rank == current_rank and candidate_entity != current_entity:
            return True

    return False


def dedupe_sentences(items: list[str]) -> list[str]:
    result = []
    for item in items:
        item = clean_sentence(item)
        if not item or is_redundant(item, result) or contradicts_existing(item, result):
            continue
        result.append(item)
    return result


def normalize_chart_type(chart_type: str) -> str:
    chart_type = clean_sentence(chart_type).replace("_", " ")
    chart_type = re.sub(r"\s+", " ", chart_type)

    mapping = {
        "simple bar": "bar chart",
        "simple bar chart": "bar chart",
        "grouped bar": "grouped bar chart",
        "grouped bar chart": "grouped bar chart",
        "stacked bar": "stacked bar chart",
        "stacked bar chart": "stacked bar chart",
    }
    return mapping.get(chart_type.lower(), chart_type or "chart")


def normalize_measure_text(text: str) -> str:
    text = clean_sentence(text)
    if not text:
        return ""

    replacements = {
        " among adults quartiles": " among adults",
        " quartiles": "",
        " across states": "",
    }
    lowered = text.lower()
    for old, new in replacements.items():
        if old in lowered:
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            text = pattern.sub(new, text)
            lowered = text.lower()

    return clean_sentence(text)


def detect_stacked_quartile_chart(chart_type: str, title: str) -> bool:
    title_key = clean_sentence(title).lower()
    chart_key = clean_sentence(chart_type).lower()
    return (
        "stacked" in chart_key
        or "quartile" in title_key
    ) and "distribution by" in title_key and "across states" in title_key


def rewrite_title_intro(chart_type: str, title: str) -> str:
    normalized_type = normalize_chart_type(chart_type)
    title = clean_sentence(title)
    title_key = title.lower()

    top_match = re.match(r"top\s+\d+\s+counties\s+for\s+(.+)", title, flags=re.IGNORECASE)
    if top_match:
        measure = normalize_measure_text(top_match.group(1))
        return f"This {normalized_type} shows the counties with the highest {measure}."

    comparison_match = re.match(
        r"county comparison:\s*(.+?)\s+vs\.?\s+(.+)",
        title,
        flags=re.IGNORECASE,
    )
    if comparison_match:
        left = clean_sentence(comparison_match.group(1))
        right = clean_sentence(comparison_match.group(2))
        return f"This grouped bar chart compares {left} and {right} across counties."

    distribution_match = re.match(
        r"county distribution by\s+(.+?)\s+quartiles across states",
        title,
        flags=re.IGNORECASE,
    )
    if distribution_match:
        measure = normalize_measure_text(distribution_match.group(1))
        return (
            f"This stacked bar chart shows how counties are distributed across states "
            f"by quartiles of {measure}."
        )

    stacked_distribution_match = re.match(
        r"county distribution by\s+(.+?)\s+across states",
        title,
        flags=re.IGNORECASE,
    )
    if stacked_distribution_match and "stacked" in clean_sentence(chart_type).lower():
        measure = normalize_measure_text(stacked_distribution_match.group(1))
        return (
            f"This stacked bar chart shows how counties are distributed across states "
            f"for {measure}."
        )

    if title_key and normalized_type != "chart":
        return f"This {normalized_type} shows {title.lower()}."
    if title_key:
        return f"This chart shows {title.lower()}."
    if normalized_type != "chart":
        return f"This {normalized_type} shows the data."
    return "This chart shows the data."


def normalize_axis_label(label: str, unit: str) -> str:
    label = clean_sentence(label)
    unit = clean_sentence(unit).lower()
    lowered = label.lower()

    if not label:
        return ""
    if lowered == "data value":
        if unit == "percent":
            return "percentage values"
        return ""
    if lowered in {"county count", "count of counties", "count"}:
        return "the number of counties"
    if lowered in {"percentage", "percent", "percentages"}:
        return "percentage values"
    if lowered.endswith(" (%)") or "%" in lowered:
        return "percentage values"
    return label.lower()


def build_axis_sentence(axes: dict, chart_type: str, title: str) -> str:
    x_axis = clean_sentence(axes.get("x_axis_title", ""))
    y_axis = clean_sentence(axes.get("y_axis_title", ""))
    unit = clean_sentence(axes.get("y_axis_unit", ""))

    if detect_stacked_quartile_chart(chart_type, title):
        return "States are shown on the x-axis, and the y-axis shows the number of counties."

    x_key = x_axis.lower()
    y_phrase = normalize_axis_label(y_axis, unit)
    unit_key = unit.lower()

    if y_phrase == "percentage values" or unit_key == "percent":
        if x_key == "county":
            return "Counties are shown on the x-axis, and values are shown as percentages."
        if x_key == "state":
            return "States are shown on the x-axis, and values are shown as percentages."
        return "Values are shown as percentages."

    if x_key == "county" and not y_phrase:
        return "Counties are shown on the x-axis."
    if x_key == "state" and y_phrase:
        return f"States are shown on the x-axis, and the y-axis shows {y_phrase}."
    if x_key == "county" and y_phrase:
        return f"Counties are shown on the x-axis, and the y-axis shows {y_phrase}."
    if y_phrase:
        return f"The y-axis shows {y_phrase}."
    return ""


def polish_sentence(text: str, chart_type: str, title: str) -> str:
    text = clean_sentence(text)
    if not text:
        return ""

    if detect_stacked_quartile_chart(chart_type, title):
        text = re.sub(r"\bhighest count\b", "highest total county count", text, flags=re.IGNORECASE)
        text = re.sub(r"\blowest count\b", "lowest total county count", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhighest counts\b", "largest total county counts", text, flags=re.IGNORECASE)
        text = re.sub(r"\blowest counts\b", "smallest total county counts", text, flags=re.IGNORECASE)
        text = re.sub(r"\ba count between\b", "a total county count between", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\bhighest counts of ([^.]+)",
            r"largest total county counts in this chart",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"\bData Value\b", "value", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCounty Count\b", "county count", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+;", ";", text)

    return ensure_period(text)


def gather_content_sentences(data: dict) -> list[str]:
    chart_type = data.get("chart_type", "")
    title = data.get("title", "")

    buckets = [
        dedupe_sentences(data.get("key_values", []) or [])[:3],
        dedupe_sentences(data.get("comparisons", []) or [])[:2],
        dedupe_sentences(data.get("trends", []) or [])[:1],
    ]

    takeaway = clean_sentence(data.get("takeaway", ""))
    if takeaway:
        buckets.append([takeaway])

    selected = []
    for bucket in buckets:
        for item in bucket:
            if is_redundant(item, selected) or contradicts_existing(item, selected):
                continue
            selected.append(item)

    polished = []
    for item in selected:
        sentence = polish_sentence(item, chart_type, title)
        if sentence and not is_redundant(sentence, polished):
            polished.append(sentence)
        if len(polished) >= 4:
            return polished

    return polished


def render_natural(data: dict) -> str:
    chart_type = data.get("chart_type", "")
    title = data.get("title", "")
    axes = data.get("axes", {}) or {}

    sentences = []

    intro = rewrite_title_intro(chart_type, title)
    if intro:
        sentences.append(intro)

    axis_sentence = build_axis_sentence(axes, chart_type, title)
    if axis_sentence and not is_redundant(axis_sentence, sentences):
        sentences.append(axis_sentence)

    for sentence in gather_content_sentences(data):
        if not is_redundant(sentence, sentences) and not contradicts_existing(sentence, sentences):
            sentences.append(sentence)

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
