"""Turn the geometric margin into a calibrated confidence and score its reliability.

	Step 2 of Experiment 2. Reads the per-point margins from step 1, rank-normalizes each margin
	to a confidence in [0, 1] (margins are unbounded and their scale differs by model and layer),
	bins the confidence, and compares the mean confidence in each bin to the empirical accuracy in
	that bin. A well-calibrated margin sits on the diagonal. Reports the expected calibration error
	(ECE) per (space, layer, model) and a pooled reliability curve per (space, model).

	python step2_margin_calibration.py
	python step2_margin_calibration.py --space rawpca --models power_multi qda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.space import OUTPUT_DIR as DATA_DIR, FIGURES_DIR

DEFAULT_MODELS = ["power_multi", "qda", "lda"]
BINS = 10


# Rank-normalize a margin column to a confidence in [0, 1].
def to_confidence(margin: np.ndarray) -> np.ndarray:
	if len(margin) <= 1:
		return np.zeros_like(margin, dtype=float)
	order = margin.argsort().argsort()
	return order / (len(margin) - 1)


# Bin confidence against correctness and return per-bin mean confidence, accuracy, and count.
def reliability(confidence: np.ndarray, correct: np.ndarray):
	edges = np.linspace(0.0, 1.0, BINS + 1)
	index = np.clip(np.digitize(confidence, edges[1:-1]), 0, BINS - 1)
	conf_bin = []
	acc_bin = []
	count_bin = []
	for b in range(BINS):
		mask = index == b
		if not mask.any():
			conf_bin.append(np.nan)
			acc_bin.append(np.nan)
			count_bin.append(0)
			continue
		conf_bin.append(float(confidence[mask].mean()))
		acc_bin.append(float(correct[mask].mean()))
		count_bin.append(int(mask.sum()))
	return np.array(conf_bin), np.array(acc_bin), np.array(count_bin)


# Expected calibration error: count-weighted mean gap between confidence and accuracy.
def expected_calibration_error(conf_bin, acc_bin, count_bin) -> float:
	valid = count_bin > 0
	if not valid.any():
		return float("nan")
	weights = count_bin[valid] / count_bin[valid].sum()
	return float(np.sum(weights * np.abs(conf_bin[valid] - acc_bin[valid])))


# Load every margins CSV for one space and model across the requested layers.
def load_margins(data_dir: Path, space: str, model: str, layers):
	frames = []
	for layer in layers:
		path = data_dir / f"margins_{space}_layer{layer}_{model}.csv"
		if path.exists():
			frame = pd.read_csv(path)
			frame["layer"] = layer
			frames.append(frame)
	return frames


# Draw a pooled reliability curve for one space and model.
def plot_reliability(space: str, model: str, confidence, correct, out_path: Path) -> None:
	conf_bin, acc_bin, count_bin = reliability(confidence, correct)
	ece = expected_calibration_error(conf_bin, acc_bin, count_bin)
	fig, ax = plt.subplots(figsize=(6.0, 6.0))
	ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
	valid = count_bin > 0
	ax.plot(conf_bin[valid], acc_bin[valid], marker="o", color="#1ecb96", label=f"{model} (ECE {ece:.3f})")
	ax.set_xlabel("margin confidence (rank-normalized)")
	ax.set_ylabel("empirical accuracy")
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_title(f"Margin reliability, {space} / {model}")
	ax.legend(loc="upper left", fontsize=9)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments, compute per-layer ECE, and draw pooled reliability curves.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
	parser.add_argument("--space", choices=["rawpca", "pns"], default="rawpca")
	parser.add_argument("--layers", type=int, nargs="+", default=None)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	summary = pd.read_csv(args.data_dir / "boundary_summary.csv")
	layers = args.layers if args.layers else sorted(summary["layer"].unique().tolist())
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	rows = []
	for model in args.models:
		frames = load_margins(args.data_dir, args.space, model, layers)
		if not frames:
			print(f"no margins for {args.space}/{model}", file=sys.stderr)
			continue
		for frame in frames:
			confidence = to_confidence(frame["margin"].to_numpy())
			conf_bin, acc_bin, count_bin = reliability(confidence, frame["correct"].to_numpy())
			rows.append({
				"space": args.space,
				"layer": int(frame["layer"].iloc[0]),
				"model": model,
				"n": len(frame),
				"ece": expected_calibration_error(conf_bin, acc_bin, count_bin),
			})
		pooled = pd.concat(frames, ignore_index=True)
		confidence = to_confidence(pooled["margin"].to_numpy())
		plot_reliability(args.space, model, confidence, pooled["correct"].to_numpy(), args.figures_dir / f"reliability_{args.space}_{model}.png")

	calibration = pd.DataFrame(rows)
	out_path = args.data_dir / f"calibration_{args.space}.csv"
	calibration.to_csv(out_path, index=False)
	print(f"wrote {out_path} ({len(calibration)} rows)", file=sys.stderr)
	print(calibration.to_string(index=False))


if __name__ == "__main__":
	main()
