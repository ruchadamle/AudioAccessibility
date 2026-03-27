import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_runtime_prompt(base_prompt: str, row: pd.Series) -> str:
    chart_id = row["chart_id"]
    chart_type = row["chart_type"]
    measures = row["measures"]

    runtime_header = (
        f"Chart ID: {chart_id}\n"
        f"Expected chart subtype: {chart_type}\n"
        f"Measure(s): {measures}\n\n"
    )
    return runtime_header + base_prompt


class QwenCaptioner:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model_id = model_id
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate_from_image(self, image_path: str, prompt_text: str, max_new_tokens: int = 512) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return output_text.strip()


class LlavaCaptioner:
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf"):
        self.model_id = model_id
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    def generate_from_image(self, image_path: str, prompt_text: str, max_new_tokens: int = 512) -> str:
        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        # Move tensors to the model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_token_count = inputs["input_ids"].shape[1]
        generated = output[:, input_token_count:]
        output_text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return output_text.strip()


def ensure_dirs(root: Path) -> None:
    (root / "baseline").mkdir(parents=True, exist_ok=True)
    (root / "hierarchical_raw").mkdir(parents=True, exist_ok=True)
    (root / "hierarchical_json").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def try_parse_json(text: str):
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw_output": text}, False


def run_generation(
    manifest_path: str,
    baseline_prompt_path: str,
    hierarchical_prompt_path: str,
    output_root: str,
    model_name: str,
    limit: int | None = None,
    start_idx: int = 0,
):
    df = pd.read_csv(manifest_path)

    if limit is not None:
        df = df.iloc[start_idx:start_idx + limit]
    else:
        df = df.iloc[start_idx:]

    baseline_prompt = load_text(baseline_prompt_path)
    hierarchical_prompt = load_text(hierarchical_prompt_path)

    output_root = Path(output_root)
    ensure_dirs(output_root)

    if model_name == "qwen":
        captioner = QwenCaptioner()
        actual_model_id = captioner.model_id
    elif model_name == "llava":
        captioner = LlavaCaptioner()
        actual_model_id = captioner.model_id
    else:
        raise ValueError("model_name must be 'qwen' or 'llava'")

    run_records = []

    for _, row in df.iterrows():
        chart_id = row["chart_id"]
        image_path = row["png_path"]

        print(f"Processing {chart_id} with {actual_model_id}...")

        baseline_runtime_prompt = build_runtime_prompt(baseline_prompt, row)
        hierarchical_runtime_prompt = build_runtime_prompt(hierarchical_prompt, row)

        # Baseline caption
        baseline_output = captioner.generate_from_image(
            image_path=image_path,
            prompt_text=baseline_runtime_prompt,
            max_new_tokens=300,
        )
        save_text(output_root / "baseline" / f"{chart_id}.txt", baseline_output)

        # Hierarchical output
        hierarchical_output = captioner.generate_from_image(
            image_path=image_path,
            prompt_text=hierarchical_runtime_prompt,
            max_new_tokens=500,
        )
        save_text(output_root / "hierarchical_raw" / f"{chart_id}.txt", hierarchical_output)

        parsed_json, valid_json = try_parse_json(hierarchical_output)
        with open(output_root / "hierarchical_json" / f"{chart_id}.json", "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)

        run_records.append(
            {
                "chart_id": chart_id,
                "chart_type": row["chart_type"],
                "measures": row["measures"],
                "image_path": image_path,
                "model_name": model_name,
                "model_id": actual_model_id,
                "baseline_file": str(output_root / "baseline" / f"{chart_id}.txt"),
                "hierarchical_raw_file": str(output_root / "hierarchical_raw" / f"{chart_id}.txt"),
                "hierarchical_json_file": str(output_root / "hierarchical_json" / f"{chart_id}.json"),
                "hierarchical_valid_json": valid_json,
            }
        )

    pd.DataFrame(run_records).to_csv(output_root / "metadata" / "run_manifest.csv", index=False)
    print(f"Done. Outputs saved under: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="manifest.csv")
    parser.add_argument("--baseline_prompt", type=str, default="prompts/baseline_v1.txt")
    parser.add_argument("--hierarchical_prompt", type=str, default="prompts/hierarchical_v2.txt")
    parser.add_argument("--output_root", type=str, default="outputs/qwen_one_pass")
    parser.add_argument("--model", type=str, choices=["qwen", "llava"], default="qwen")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    run_generation(
        manifest_path=args.manifest,
        baseline_prompt_path=args.baseline_prompt,
        hierarchical_prompt_path=args.hierarchical_prompt,
        output_root=args.output_root,
        model_name=args.model,
        limit=args.limit,
        start_idx=args.start_idx,
    )