#!/usr/bin/env python3
"""Load four jailbreak / content-safety benchmarks into one common schema.

	SORRY-Bench     sorry-bench/sorry-bench-202503                 ~440 unsafe instructions (gated)
	WildJailbreak   allenai/wildjailbreak                          ~261,534 rows (gated)
	ToxicChat       lmsys/toxic-chat (toxicchat0124)               10,165 rows
	Aegis 2.0       nvidia/Aegis-AI-Content-Safety-Dataset-2.0     33,416 rows

All four load from the Hugging Face Hub via the `datasets` package. SORRY-Bench,
WildJailbreak, and Aegis require accepting the dataset terms and an HF token.

	python load_datasets.py
	python load_datasets.py --datasets toxic-chat aegis-2.0
	python load_datasets.py --export jsonl

	from load_datasets import load_all
	corpora = load_all()
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

DATA_DIR = Path(__file__).resolve().parent / "data"
NORMALISED_DIR = DATA_DIR / "normalised"


# One normalised prompt, shared across every dataset.
@dataclass
class Example:
	id: str
	prompt: str
	dataset: str
	subset: str
	split: str
	label: str
	category: str
	meta: dict = field(default_factory=dict)


# Build a short, stable id from the given parts.
def make_id(*parts: object) -> str:
	digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8"))
	return digest.hexdigest()[:12]


# Return the datasets.load_dataset callable, or raise a clear error if it is missing.
def require_datasets() -> Callable:
	try:
		from datasets import load_dataset
	except ImportError as error:
		raise RuntimeError(
			"These loaders need the 'datasets' package: pip install datasets"
		) from error
	return load_dataset


# Load SORRY-Bench 2025/03, keeping only the base (unmutated) prompt of each row.
def load_sorry_bench() -> list[Example]:
	load_dataset = require_datasets()
	print("loading sorry-bench/sorry-bench-202503 (default/train)", file=sys.stderr)
	rows = load_dataset("sorry-bench/sorry-bench-202503", "default", split="train")
	examples: list[Example] = []
	for index, row in enumerate(rows):
		style = row.get("prompt_style")
		if style is not None and style != "base":
			continue
		turns = row.get("turns")
		if isinstance(turns, list):
			prompt = " ".join(str(part) for part in turns if part).strip()
		else:
			prompt = str(turns).strip() if turns else ""
		if not prompt:
			continue
		category_value = row.get("category")
		label = str(category_value) if category_value is not None else "harmful"
		examples.append(Example(
			id=make_id("sorry-bench", index),
			prompt=prompt,
			dataset="sorry-bench",
			subset="default",
			split="train",
			label=label,
			category="harmful",
			meta={
				"question_id": row.get("question_id"),
				"category_id": category_value,
				"prompt_style": style,
			},
		))
	print(f"sorry-bench: kept {len(examples)} base prompts", file=sys.stderr)
	return examples


# Load the WildJailbreak train config (~262k gated rows) with vanilla/adversarial prompts.
def load_wildjailbreak() -> list[Example]:
	load_dataset = require_datasets()
	print("loading allenai/wildjailbreak (train)", file=sys.stderr)
	try:
		rows = load_dataset("allenai/wildjailbreak", "train", split="train")
	except Exception as error:
		raise RuntimeError(
			"Failed to load allenai/wildjailbreak. This dataset is gated: accept the "
			"terms at https://huggingface.co/datasets/allenai/wildjailbreak and set an "
			"HF token (run `huggingface-cli login` or set HUGGINGFACE_HUB_TOKEN)."
		) from error
	examples: list[Example] = []
	for index, row in enumerate(rows):
		data_type = (row.get("data_type") or "").strip()
		vanilla = (row.get("vanilla") or "").strip()
		adversarial = (row.get("adversarial") or "").strip()
		completion = (row.get("completion") or "").strip()
		is_adversarial = data_type.startswith("adversarial")
		prompt = adversarial if (is_adversarial and adversarial) else vanilla
		if not prompt:
			continue
		category = "harmful" if data_type.endswith("harmful") else "benign"
		examples.append(Example(
			id=make_id("wildjailbreak", index),
			prompt=prompt,
			dataset="wildjailbreak",
			subset="train",
			split="train",
			label=data_type,
			category=category,
			meta={
				"data_type": data_type,
				"is_adversarial": is_adversarial,
				"has_completion": bool(completion),
			},
		))
		if (index + 1) % 20000 == 0:
			print(f"wildjailbreak: processed {index + 1} rows", file=sys.stderr)
	print(f"wildjailbreak: loaded {len(examples)} examples", file=sys.stderr)
	return examples


# Load the ToxicChat dataset (toxicchat0124 config, train and test splits).
def load_toxic_chat() -> list[Example]:
	load_dataset = require_datasets()
	examples: list[Example] = []
	for split in ("train", "test"):
		print(f"loading lmsys/toxic-chat toxicchat0124 ({split})", file=sys.stderr)
		rows = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split)
		for index, row in enumerate(rows):
			prompt = row.get("user_input")
			if not prompt:
				continue
			toxicity = int(row.get("toxicity") or 0)
			jailbreaking = int(row.get("jailbreaking") or 0)
			category = "toxic" if toxicity == 1 else "benign"
			label = "toxic" if toxicity == 1 else "non_toxic"
			conv_id = row.get("conv_id") or index
			examples.append(Example(
				id=make_id("toxic-chat", split, conv_id),
				prompt=prompt,
				dataset="toxic-chat",
				subset="toxicchat0124",
				split=split,
				label=label,
				category=category,
				meta={"jailbreaking": jailbreaking, "toxicity": toxicity},
			))
	print(f"toxic-chat: loaded {len(examples)} examples", file=sys.stderr)
	return examples


# Load the NVIDIA Aegis 2.0 content-safety dataset across all splits.
def load_aegis() -> list[Example]:
	load_dataset = require_datasets()
	examples: list[Example] = []
	for split in ("train", "validation", "test"):
		print(f"loading nvidia/Aegis-AI-Content-Safety-Dataset-2.0 ({split})", file=sys.stderr)
		try:
			rows = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "default", split=split)
		except Exception as error:
			raise RuntimeError(
				"Failed to load nvidia/Aegis-AI-Content-Safety-Dataset-2.0. If access is "
				"restricted, accept the terms at https://huggingface.co/datasets/"
				"nvidia/Aegis-AI-Content-Safety-Dataset-2.0 and set an HF token."
			) from error
		for index, row in enumerate(rows):
			prompt = row.get("prompt")
			if not prompt:
				continue
			label = row.get("prompt_label") or ""
			category = "harmful" if label == "unsafe" else "benign"
			examples.append(Example(
				id=make_id("aegis", split, index),
				prompt=prompt,
				dataset="aegis-2.0",
				subset="default",
				split=split,
				label=label,
				category=category,
				meta={
					"source_id": row.get("id"),
					"violated_categories": row.get("violated_categories"),
					"response": row.get("response"),
					"response_label": row.get("response_label"),
					"prompt_label_source": row.get("prompt_label_source"),
					"response_label_source": row.get("response_label_source"),
				},
			))
	print(f"aegis-2.0: loaded {len(examples)} examples", file=sys.stderr)
	return examples


LOADERS: dict[str, Callable[[], list[Example]]] = {
	"sorry-bench": load_sorry_bench,
	"wildjailbreak": load_wildjailbreak,
	"toxic-chat": load_toxic_chat,
	"aegis-2.0": load_aegis,
}


# Load the requested datasets (default: all) as {name: list[Example]}.
def load_all(datasets: list[str] | None = None) -> dict[str, list[Example]]:
	names = datasets or list(LOADERS)
	corpora: dict[str, list[Example]] = {}
	for name in names:
		if name not in LOADERS:
			raise KeyError(f"Unknown dataset {name!r}; choose from {list(LOADERS)}")
		print(f"Loading {name} ...", file=sys.stderr)
		corpora[name] = LOADERS[name]()
	return corpora


# Format a per-dataset table of counts broken down by split and label.
def format_summary(corpora: dict[str, list[Example]]) -> str:
	header = f"{'dataset':<28} {'split':<12} {'label':<13} {'count':>7}"
	lines = [header, "-" * len(header)]
	total = 0
	for name, examples in corpora.items():
		counts: dict[tuple[str, str], int] = {}
		for example in examples:
			key = (example.split, example.label)
			counts[key] = counts.get(key, 0) + 1
		for (split, label), count in sorted(counts.items()):
			lines.append(f"{name:<28} {split:<12} {label:<13} {count:>7}")
			total += count
	lines.append("-" * len(header))
	lines.append(f"{'TOTAL':<54} {total:>7}")
	return "\n".join(lines)


# Write each dataset's normalised records to data/normalised/<name>.jsonl.
def export_jsonl(corpora: dict[str, list[Example]]) -> None:
	NORMALISED_DIR.mkdir(parents=True, exist_ok=True)
	for name, examples in corpora.items():
		out_path = NORMALISED_DIR / f"{name}.jsonl"
		with out_path.open("w", encoding="utf-8") as handle:
			for example in examples:
				handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
		print(f"  wrote {out_path} ({len(examples)} records)", file=sys.stderr)


# Parse arguments, load the datasets, print the summary, optionally export.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument(
		"--datasets", nargs="+", choices=list(LOADERS),
		help="Subset of datasets to load (default: all).",
	)
	parser.add_argument(
		"--export", choices=["jsonl"],
		help="Also write normalised records to data/normalised/<name>.jsonl",
	)
	args = parser.parse_args()

	corpora = load_all(args.datasets)
	print(format_summary(corpora))
	if args.export == "jsonl":
		export_jsonl(corpora)


if __name__ == "__main__":
	main()
