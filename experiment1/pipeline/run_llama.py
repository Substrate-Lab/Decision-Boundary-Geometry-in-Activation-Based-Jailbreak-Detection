#!/usr/bin/env python3
"""Collect Llama activations at the later layers, without touching the Qwen run.

	step1_collect_activations.py writes to fixed filenames under experiment1/data --
	labels.csv, labels_full.csv, collection_meta.json, activations_layer<N>.npy -- with no model
	in any path. Pointing it at a second model would overwrite the committed Qwen outputs in
	place, and because the layer CSVs carry no model column the result would be unattributable.

	This runner avoids that instead of asking you to remember it. It redirects the collection
	step's output directory to

		experiment1/data/runs/<model-slug>/

	by rebinding step1's DATA_DIR before calling it, so the existing flat files are never opened
	for writing. Nothing in the committed Qwen data is read, moved, or modified.

	Layers default to the back half of the network, sampled every other block. Experiment 1's
	reported Qwen result was layer 30 of 36 (0.83 of depth), so the region worth resolving on a
	new model is the deep end, not a spread across the whole stack.

	Preflight first, on the machine that will do the run:

		python experiment1/pipeline/run_llama.py --dry-run

	That checks CUDA, VRAM, quantisation support, the HF token, model access, and disk, and
	prints the exact collection it would perform -- without loading the model. Then:

		python experiment1/pipeline/run_llama.py

	Defaults reproduce the Qwen collection's settings (500/class, pools 1400/700, batch 4,
	24 new tokens, 4-bit, seed 0) so the two runs are comparable.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PIPELINE_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Settings the committed Qwen run used, so a Llama run is comparable rather than merely similar.
QWEN_SETTINGS = {
	"per_class": 500,
	"harmful_pool": 1400,
	"benign_pool": 700,
	"batch_size": 4,
	"max_new_tokens": 24,
}


# Filesystem-safe short name for a model id or local path.
def model_slug(model_id: str) -> str:
	return re.split(r"[\\/]", str(model_id).strip())[-1]


# Report one preflight result and remember whether it blocks the run.
class Preflight:
	def __init__(self):
		self.blocking = []
		self.warnings = []

	def check(self, name, ok, detail="", blocking=True):
		mark = "PASS" if ok else ("FAIL" if blocking else "WARN")
		print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
		if not ok:
			(self.blocking if blocking else self.warnings).append(name)
		return ok


# GPU, quantisation, credentials, model access and disk, none of which need the weights.
def preflight(model_id: str, want_4bit: bool, data_dir: Path) -> Preflight:
	report = Preflight()
	print("preflight")

	try:
		import torch
	except ImportError:
		report.check("torch is installed", False, "pip install torch (CUDA build)")
		return report
	report.check("torch is installed", True, torch.__version__)

	has_cuda = torch.cuda.is_available()
	report.check("CUDA device is visible", has_cuda,
	             torch.cuda.get_device_name(0) if has_cuda else "collection needs a GPU")
	vram = 0.0
	if has_cuda:
		vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
		report.check("VRAM is sufficient", vram >= 7.0, f"{vram:.1f} GB", blocking=False)

	if want_4bit:
		report.check("bitsandbytes is available for 4-bit",
		             importlib.util.find_spec("bitsandbytes") is not None,
		             "pip install bitsandbytes, or pass --no-4bit")

	try:
		from transformers import AutoConfig
	except ImportError:
		report.check("transformers is installed", False, "pip install transformers")
		return report
	report.check("transformers is installed", True)

	# A gated repo fails here, on config.json, long before any weight is fetched.
	depth = None
	try:
		config = AutoConfig.from_pretrained(model_id)
		depth = int(getattr(config, "num_hidden_layers", 0))
		hidden = int(getattr(config, "hidden_size", 0))
		report.check("model config is reachable", True,
		             f"{depth} layers, hidden {hidden}")
	except Exception as error:
		message = str(error).splitlines()[0][:120]
		report.check("model config is reachable", False, message)
		print("         gated repo? accept the terms on the model page, then "
		      "`huggingface-cli login` or set HF_TOKEN.")

	free_gb = shutil.disk_usage(data_dir.parent if data_dir.exists() else REPO_ROOT).free / 1024 ** 3
	report.check("disk space for weights + activations", free_gb >= 40.0,
	             f"{free_gb:.0f} GB free; 8B weights ~16 GB plus activations",
	             blocking=False)

	report.check("datasets is installed",
	             importlib.util.find_spec("datasets") is not None, "pip install datasets")

	report.depth = depth
	report.vram = vram
	return report


# The back half of the network, every other block, which is where the Qwen signal peaked.
def later_layers(depth: int, stride: int = 2) -> list:
	start = max(1, depth // 2)
	layers = list(range(start, depth + 1, stride))
	if layers and layers[-1] != depth:
		layers.append(depth)
	return layers


# Import step1 by path and rebind its output directory before it writes anything.
def run_collection(model_id: str, layers: list, data_dir: Path, args) -> int:
	spec = importlib.util.spec_from_file_location(
		"step1_collect_activations", PIPELINE_DIR / "step1_collect_activations.py"
	)
	step1 = importlib.util.module_from_spec(spec)
	# Register before executing: @dataclass resolves its own module via sys.modules, so a
	# module_from_spec that is not registered raises AttributeError on step1's Config.
	sys.modules[spec.name] = step1
	spec.loader.exec_module(step1)

	# DATA_DIR is looked up as a module global at call time, so rebinding it here redirects
	# every write -- the labels CSVs, the .npy arrays, collection_meta.json and the analytics
	# run report -- into the per-model directory.
	data_dir.mkdir(parents=True, exist_ok=True)
	step1.DATA_DIR = data_dir

	argv = [
		"step1_collect_activations.py",
		"--model", model_id,
		"--layers", ",".join(str(layer) for layer in layers),
		"--per-class", str(args.per_class),
		"--harmful-pool", str(args.harmful_pool),
		"--benign-pool", str(args.benign_pool),
		"--batch-size", str(args.batch_size),
		"--max-new-tokens", str(args.max_new_tokens),
		"--pooling", args.pooling,
	]
	if args.no_4bit:
		argv.append("--no-4bit")
	if args.safety_prompt:
		argv.append("--safety-prompt")

	print(f"\ncollecting into {data_dir}")
	print("  " + " ".join(argv[1:]) + "\n")
	saved = sys.argv
	try:
		sys.argv = argv
		step1.main()
	finally:
		sys.argv = saved
	return 0


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--model", default=DEFAULT_MODEL, help="HF id or local path.")
	parser.add_argument("--layers", type=str, default=None,
	                    help="Comma-separated override; default is the back half, stride 2.")
	parser.add_argument("--stride", type=int, default=2, help="Spacing of the sampled layers.")
	parser.add_argument("--per-class", type=int, default=QWEN_SETTINGS["per_class"])
	parser.add_argument("--harmful-pool", type=int, default=QWEN_SETTINGS["harmful_pool"])
	parser.add_argument("--benign-pool", type=int, default=QWEN_SETTINGS["benign_pool"])
	parser.add_argument("--batch-size", type=int, default=QWEN_SETTINGS["batch_size"])
	parser.add_argument("--max-new-tokens", type=int, default=QWEN_SETTINGS["max_new_tokens"])
	parser.add_argument("--pooling", choices=["last", "mean"], default="last")
	parser.add_argument("--no-4bit", action="store_true")
	parser.add_argument("--safety-prompt", action="store_true")
	parser.add_argument("--dry-run", action="store_true",
	                    help="Run every check and print the plan, without loading the model.")
	args = parser.parse_args()

	slug = model_slug(args.model)
	data_dir = EXPERIMENT_DIR / "data" / "runs" / slug
	print(f"model      {args.model}")
	print(f"output     {data_dir.relative_to(REPO_ROOT)}")
	print("protected  experiment1/data/*.csv and collection_meta.json are never written\n")

	report = preflight(args.model, not args.no_4bit, data_dir)
	depth = getattr(report, "depth", None)

	if args.layers:
		layers = [int(part) for part in args.layers.split(",") if part.strip()]
		source = "explicit --layers"
	elif depth:
		layers = later_layers(depth, args.stride)
		source = f"back half of {depth} layers, stride {args.stride}"
	else:
		layers = []
		source = "unavailable (model config unreachable)"

	print(f"\nlayers     {source}")
	if layers:
		print(f"           {layers}  ({len(layers)} layers)")
		if depth:
			print(f"           deepest sampled = {max(layers) / depth:.2f} of depth; "
			      f"Qwen's reported best was 30/36 = 0.83")

	if data_dir.exists() and any(data_dir.glob("activations_layer*.npy")):
		print(f"\nnote: {data_dir.relative_to(REPO_ROOT)} already holds activations; "
		      f"re-running overwrites that model's own outputs only.")

	if report.blocking:
		print(f"\nBLOCKED: {', '.join(report.blocking)}")
		print("Fix the FAIL items above and re-run --dry-run.")
		return 1
	if report.warnings:
		print(f"\nwarnings (not blocking): {', '.join(report.warnings)}")
	if not layers:
		print("\nBLOCKED: no layers resolved. Pass --layers explicitly, e.g. --layers 16,20,24,28,32")
		return 1

	if args.dry_run:
		print("\ndry run only — nothing was loaded or written.")
		print("Re-run without --dry-run to collect.")
		return 0

	return run_collection(args.model, layers, data_dir, args)


if __name__ == "__main__":
	raise SystemExit(main())
