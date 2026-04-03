from __future__ import annotations

import math
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
RANDOM_SEED = 20260321
TOTAL_CHARTS = 90
SIMPLE_CHARTS = 36
GROUPED_CHARTS = 27
STACKED_CHARTS = 27
PNG_DIR = REPO_ROOT / "charts" / "png"
SVG_DIR = REPO_ROOT / "charts" / "svg"
MANIFEST_PATH = REPO_ROOT / "manifest.csv"
CSV_GLOB_PATTERNS = (
    "PLACES__Local_Data_for_Better_Health,_County_Data,_*.csv",
    "places__localdata*.csv",
    "*qaee-pcj3*.csv",
)


def wrap_title(title: str, width: int = 70) -> str:
    return "\n".join(textwrap.wrap(title, width=width))


def sanitize_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "chart"


def discover_csv_file() -> Path:
    for pattern in CSV_GLOB_PATTERNS:
        for search_dir in (BASE_DIR, REPO_ROOT):
            matches = sorted(search_dir.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError("Could not find the local CDC PLACES CSV file in the script directory.")


def load_data() -> pd.DataFrame:
    csv_path = discover_csv_file()
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize the local PLACES schema to the requested column name
    if "CountyName" not in df.columns and "LocationName" in df.columns:
        df = df.rename(columns={"LocationName": "CountyName"})

    required_columns = ["MeasureId", "Measure", "Data_Value", "CountyName", "StateAbbr"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()
    df["MeasureId"] = df["MeasureId"].astype(str).str.strip()
    df["Measure"] = df["Measure"].astype(str).str.strip()
    df["CountyName"] = df["CountyName"].astype(str).str.strip()
    df["StateAbbr"] = df["StateAbbr"].astype(str).str.strip()
    df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")

    df = df.dropna(subset=["MeasureId", "Measure", "CountyName", "StateAbbr", "Data_Value"])
    df = df[df["CountyName"].ne("")]
    df = df[df["StateAbbr"].ne("")]
    df = df[df["MeasureId"].ne("")]
    df = df[df["Measure"].ne("")]

    # Keep one value per county-state-measure in case the source contains duplicates.
    # Collapse any duplicate county-measure rows into one numeric value
    df = (
        df.groupby(["MeasureId", "Measure", "CountyName", "StateAbbr"], as_index=False)["Data_Value"]
        .mean()
        .sort_values(["MeasureId", "StateAbbr", "CountyName"])
        .reset_index(drop=True)
    )

    return df


def save_figure(fig: plt.Figure, chart_id: str) -> tuple[str, str]:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)

    png_path = PNG_DIR / f"{chart_id}.png"
    svg_path = SVG_DIR / f"{chart_id}.svg"

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    return f"charts/png/{chart_id}.png", f"charts/svg/{chart_id}.svg"


def generate_simple_bar(df: pd.DataFrame, rng: random.Random, chart_id: str) -> dict | None:
    measure_counts = df["MeasureId"].value_counts()
    eligible_measures = measure_counts[measure_counts >= 10].index.tolist()
    if not eligible_measures:
        return None

    measure_id = rng.choice(eligible_measures)
    subset = df[df["MeasureId"] == measure_id].copy()
    measure_name = subset["Measure"].iloc[0]
    top_n = rng.randint(6, 10)

    subset["CountyLabel"] = subset["CountyName"] + ", " + subset["StateAbbr"]
    top_counties = subset.nlargest(top_n, "Data_Value").sort_values("Data_Value", ascending=False)

    if len(top_counties) < 6:
        return None

    fig, ax = plt.subplots(figsize=(max(10, top_n * 1.2), 6))
    x_positions = np.arange(len(top_counties))
    ax.bar(x_positions, top_counties["Data_Value"], color="#4472C4", edgecolor="#2F528F", linewidth=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(top_counties["CountyLabel"], rotation=35, ha="right")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("County")
    ax.set_title(wrap_title(f"Top {len(top_counties)} counties for {measure_name}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    png_path, svg_path = save_figure(fig, chart_id)
    return {
        "chart_id": chart_id,
        "chart_type": "simple_bar",
        "measures": measure_name,
        "png_path": png_path,
        "svg_path": svg_path,
    }


def generate_grouped_bar(df: pd.DataFrame, rng: random.Random, chart_id: str) -> dict | None:
    measure_frames = {}
    for measure_id, subset in df.groupby("MeasureId"):
        indexed = subset.assign(CountyKey=subset["CountyName"] + ", " + subset["StateAbbr"]).set_index("CountyKey")
        if len(indexed) >= 8:
            measure_frames[measure_id] = indexed

    # Keep only measure pairs that share enough counties for a readable comparison
    eligible_pairs = []
    measure_ids = sorted(measure_frames)
    for first_index, measure_a in enumerate(measure_ids):
        frame_a = measure_frames[measure_a]
        for measure_b in measure_ids[first_index + 1 :]:
            frame_b = measure_frames[measure_b]
            common_counties = frame_a.index.intersection(frame_b.index)
            if len(common_counties) >= 6:
                eligible_pairs.append((measure_a, measure_b, common_counties))

    if not eligible_pairs:
        return None

    measure_a, measure_b, common_counties = rng.choice(eligible_pairs)
    count = min(rng.randint(6, 8), len(common_counties))

    county_sample = pd.DataFrame(
        {
            "CountyLabel": common_counties,
            "ValueA": measure_frames[measure_a].loc[common_counties, "Data_Value"].to_numpy(),
            "ValueB": measure_frames[measure_b].loc[common_counties, "Data_Value"].to_numpy(),
        }
    )
    county_sample["CombinedScore"] = county_sample["ValueA"] + county_sample["ValueB"]
    county_sample = county_sample.nlargest(count, "CombinedScore").sort_values("CombinedScore", ascending=False)

    if len(county_sample) < 6:
        return None

    measure_name_a = measure_frames[measure_a]["Measure"].iloc[0]
    measure_name_b = measure_frames[measure_b]["Measure"].iloc[0]

    x_positions = np.arange(len(county_sample))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(county_sample) * 1.4), 6))
    ax.bar(
        x_positions - width / 2,
        county_sample["ValueA"],
        width=width,
        label=measure_name_a,
        color="#4F81BD",
        edgecolor="#385D8A",
        linewidth=0.7,
    )
    ax.bar(
        x_positions + width / 2,
        county_sample["ValueB"],
        width=width,
        label=measure_name_b,
        color="#C0504D",
        edgecolor="#963634",
        linewidth=0.7,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(county_sample["CountyLabel"], rotation=35, ha="right")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("County")
    ax.set_title(wrap_title(f"County comparison: {measure_name_a} vs {measure_name_b}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    png_path, svg_path = save_figure(fig, chart_id)
    return {
        "chart_id": chart_id,
        "chart_type": "grouped_bar",
        "measures": f"{measure_name_a} | {measure_name_b}",
        "png_path": png_path,
        "svg_path": svg_path,
    }


def generate_stacked_bar(df: pd.DataFrame, rng: random.Random, chart_id: str) -> dict | None:
    measure_counts = df["MeasureId"].value_counts()
    eligible_measures = measure_counts[measure_counts >= 40].index.tolist()
    if not eligible_measures:
        return None

    measure_id = rng.choice(eligible_measures)
    subset = df[df["MeasureId"] == measure_id].copy()
    measure_name = subset["Measure"].iloc[0]

    # Use quartiles to turn a continuous measure into stacked category counts
    try:
        subset["Quartile"] = pd.qcut(
            subset["Data_Value"],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
    except ValueError:
        return None

    subset = subset.dropna(subset=["Quartile"])
    if subset["Quartile"].nunique() < 4:
        return None

    state_counts = subset["StateAbbr"].value_counts()
    state_pool = state_counts[state_counts >= 4].index.tolist()
    if len(state_pool) < 5:
        return None

    state_count = min(rng.randint(5, 10), len(state_pool))
    selected_states = rng.sample(state_pool, k=state_count)
    state_distribution = (
        subset[subset["StateAbbr"].isin(selected_states)]
        .groupby(["StateAbbr", "Quartile"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Q1", "Q2", "Q3", "Q4"], fill_value=0)
    )

    if len(state_distribution) < 5:
        return None

    state_distribution = state_distribution.loc[state_distribution.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(10, len(state_distribution) * 1.1), 6))
    bottom = np.zeros(len(state_distribution))
    colors = ["#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]

    for quartile, color in zip(state_distribution.columns, colors):
        values = state_distribution[quartile].to_numpy()
        ax.bar(
            state_distribution.index,
            values,
            bottom=bottom,
            label=quartile,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += values

    ax.set_ylabel("County Count")
    ax.set_xlabel("State")
    ax.set_title(wrap_title(f"County distribution by {measure_name} quartiles across states"))
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(title="Quartile", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    png_path, svg_path = save_figure(fig, chart_id)
    return {
        "chart_id": chart_id,
        "chart_type": "stacked_bar",
        "measures": measure_name,
        "png_path": png_path,
        "svg_path": svg_path,
    }


def build_chart_plan() -> list[str]:
    plan = (["simple_bar"] * SIMPLE_CHARTS) + (["grouped_bar"] * GROUPED_CHARTS) + (["stacked_bar"] * STACKED_CHARTS)
    if len(plan) != TOTAL_CHARTS:
        raise ValueError("Chart plan does not match TOTAL_CHARTS.")
    return plan


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = load_data()
    chart_plan = build_chart_plan()
    # Shuffle chart types while keeping the overall mix fixed
    rng.shuffle(chart_plan)

    manifest_rows: list[dict] = []
    attempts = 0
    max_attempts = TOTAL_CHARTS * 8

    while len(manifest_rows) < TOTAL_CHARTS and attempts < max_attempts:
        chart_type = chart_plan[len(manifest_rows)]
        chart_id = f"{len(manifest_rows) + 1:03d}_{sanitize_filename(chart_type)}"

        if chart_type == "simple_bar":
            record = generate_simple_bar(df, rng, chart_id)
        elif chart_type == "grouped_bar":
            record = generate_grouped_bar(df, rng, chart_id)
        elif chart_type == "stacked_bar":
            record = generate_stacked_bar(df, rng, chart_id)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        attempts += 1
        if record is not None:
            manifest_rows.append(record)

    if len(manifest_rows) != TOTAL_CHARTS:
        raise RuntimeError(
            f"Generated {len(manifest_rows)} charts after {attempts} attempts; expected {TOTAL_CHARTS}."
        )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_PATH, index=False)


if __name__ == "__main__":
    main()
