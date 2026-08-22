#!/usr/bin/env python3
"""Load the four jailbreak/over-refusal benchmarks into one common schema.

    HarmBench           400 harmful behaviours    (320 test / 80 val)
    HarmBench human JB  ~2,000 human jailbreaks   (1,600 test / 400 val)
    AdvBench            520 harmful instructions
    OR-Bench            80k over-refusal + 600 toxic prompts
    EVOREFUSE-TEST      582 pseudo-malicious prompts

Sources download from their canonical public locations and cache under ./data.

    python load_datasets.py
    python load_datasets.py --datasets harmbench advbench
    python load_datasets.py --export jsonl

    from load_datasets import load_all
    corpora = load_all()
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
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


# Download a url into the raw cache once, then reuse the cached copy.
def download(url: str, filename: str) -> Path:
    path = RAW_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": "substrate-loader/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())
    return path


# Read a CSV into a list of row dicts.
# utf-8-sig drops a possible BOM; DictReader copes with fields spanning newlines.
def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_{split}.csv"
)
HARMBENCH_SPLIT_FILES = {"test": "test", "validation": "val"}


# Load HarmBench text behaviours: 320 test + 80 val = 400.
# All functional categories are kept unless only_category restricts to one.
def load_harmbench(only_category: str | None = None) -> list[Example]:
    examples: list[Example] = []
    for split, file_key in HARMBENCH_SPLIT_FILES.items():
        rows = read_csv_rows(download(
            HARMBENCH_URL.format(split=file_key), f"harmbench_{split}.csv"
        ))
        for row in rows:
            functional = (row.get("FunctionalCategory") or "").strip()
            if only_category and functional.lower() != only_category.lower():
                continue
            behavior = (row.get("Behavior") or "").strip()
            if not behavior:
                continue
            prompt = compose_harmbench_prompt(behavior, row.get("ContextString"))
            examples.append(Example(
                id=row.get("BehaviorID") or make_id("harmbench", behavior),
                prompt=prompt,
                dataset="harmbench",
                subset=functional or "standard",
                split=split,
                label="harmful",
                category="harmful",
                meta={
                    "behavior": behavior,
                    "semantic_category": row.get("SemanticCategory"),
                    "tags": row.get("Tags"),
                    "context_string": (row.get("ContextString") or "").strip() or None,
                },
            ))
    return examples


# Prepend a contextual behaviour's ContextString so the prompt is the real input.
def compose_harmbench_prompt(behavior: str, context_string: str | None) -> str:
    context = (context_string or "").strip()
    return f"{context}\n\n{behavior}" if context else behavior


HARMBENCH_TEMPLATES_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "baselines/human_jailbreaks/jailbreaks.py"
)
TEMPLATES_PER_BEHAVIOUR = 5
HUMAN_JAILBREAK_SEED = 0


# Read HarmBench's hand-written jailbreak templates from its source file.
# The file defines a single JAILBREAKS list literal, which we parse without executing it.
def load_jailbreak_templates() -> list[str]:
    source = download(HARMBENCH_TEMPLATES_URL, "harmbench_jailbreaks.py").read_text("utf-8")
    for node in ast.parse(source).body:
        targets = getattr(node, "targets", [])
        if any(isinstance(t, ast.Name) and t.id == "JAILBREAKS" for t in targets):
            templates = ast.literal_eval(node.value)
            return [template for template in templates if "{0}" in template]
    raise RuntimeError("JAILBREAKS list not found in HarmBench jailbreaks.py")


# Wrap each HarmBench behaviour in several human-written jailbreak templates.
# Behaviours keep their split, and 5 templates each lands the totals near 1,600/400.
def load_harmbench_human_jailbreaks() -> list[Example]:
    behaviours = load_harmbench()
    templates = load_jailbreak_templates()
    sample_size = min(TEMPLATES_PER_BEHAVIOUR, len(templates))
    rng = random.Random(HUMAN_JAILBREAK_SEED)

    examples: list[Example] = []
    for behaviour in behaviours:
        for template_index in rng.sample(range(len(templates)), sample_size):
            examples.append(Example(
                id=make_id("hj", behaviour.id, template_index),
                prompt=templates[template_index].replace("{0}", behaviour.prompt),
                dataset="harmbench_human_jailbreaks",
                subset="human_jailbreaks",
                split=behaviour.split,
                label="harmful",
                category="harmful",
                meta={
                    "behavior_id": behaviour.id,
                    "base_behavior": behaviour.prompt,
                    "template_index": template_index,
                },
            ))
    return examples


ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)


# Load AdvBench: 520 harmful instructions, each with its target completion.
def load_advbench() -> list[Example]:
    rows = read_csv_rows(download(ADVBENCH_URL, "advbench.csv"))
    examples: list[Example] = []
    for index, row in enumerate(rows):
        goal = (row.get("goal") or "").strip()
        if not goal:
            continue
        examples.append(Example(
            id=make_id("advbench", index, goal),
            prompt=goal,
            dataset="advbench",
            subset="harmful_behaviors",
            split="train",
            label="harmful",
            category="harmful",
            meta={"target": row.get("target")},
        ))
    return examples


ORBENCH_REPO = "bench-llm/or-bench"
ORBENCH_CONFIGS = {
    "or-bench-80k": "over_refusal",
    "or-bench-toxic": "toxic",
}


# Load OR-Bench from the Hugging Face Hub: 80k over-refusal + 600 toxic prompts.
def load_orbench() -> list[Example]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "OR-Bench needs the 'datasets' package: pip install datasets"
        ) from error

    examples: list[Example] = []
    for config, label in ORBENCH_CONFIGS.items():
        dataset = load_dataset(ORBENCH_REPO, config, split="train", cache_dir=str(RAW_DIR))
        for index, row in enumerate(dataset):
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue
            examples.append(Example(
                id=make_id("orbench", config, index),
                prompt=prompt,
                dataset="or-bench",
                subset=config,
                split="train",
                label=label,
                category=label,
                meta={"category_label": row.get("category")},
            ))
    return examples


EVOREFUSE_URL = (
    "https://raw.githubusercontent.com/FishT0ucher/EVOREFUSE/main/"
    "datasets/evo_test.jsonl"
)


# Load EVOREFUSE-TEST: 582 pseudo-malicious (over-refusal) instructions.
def load_evorefuse() -> list[Example]:
    path = download(EVOREFUSE_URL, "evorefuse_test.jsonl")
    examples: list[Example] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = (row.get("instruction") or row.get("prompt") or "").strip()
        if not prompt:
            continue
        examples.append(Example(
            id=make_id("evorefuse", index),
            prompt=prompt,
            dataset="evorefuse-test",
            subset="test",
            split="test",
            label="over_refusal",
            category="over_refusal",
            meta={k: v for k, v in row.items() if k not in ("instruction", "prompt")},
        ))
    return examples


LOADERS: dict[str, Callable[[], list[Example]]] = {
    "harmbench": load_harmbench,
    "harmbench_human_jailbreaks": load_harmbench_human_jailbreaks,
    "advbench": load_advbench,
    "or-bench": load_orbench,
    "evorefuse-test": load_evorefuse,
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