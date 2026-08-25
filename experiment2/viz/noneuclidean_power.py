#!/usr/bin/env python3
"""When does curvature actually move the power-diagram boundary?

	The independent variable of §5.2 is the coordinate system, so the comparison holds everything
	else fixed -- same points, same 36 sites, same class-prior targets, same Alexandrov weight
	solve -- and varies only the distance the diagram is built from:

		chordal   ||x-y||^2 - w      ambient Euclidean restricted to S^2 (Sugihara). The
		                             control: still a Euclidean diagram, drawn on a curved set.
		geodesic  arccos(x.y)^2 - w  intrinsic, genuinely non-Euclidean.
		tangent   ||u-v||^2 - w      Euclidean in log-map coordinates at the Karcher mean --
		                             the §5.2 route that keeps boundaries exact hyperplanes.

	Chordal and geodesic coincide at w = 0 (both monotone in x.y); only solved weights separate
	them, because the weight enters on a different scale in each.

	The thing worth knowing is not "do they differ" but "when". Curvature can only act if the
	classes span enough of the sphere to be curved, so the answer is a function of how wide a cap
	the data occupies. The sweep panel measures exactly that, and marks where Experiment 1's
	calibrated operating point falls on it.

	Read the caveat before quoting the sweep: this is a 3-dimensional stand-in. Reproducing
	Experiment 1's 36-site accuracy in 3D forces the classes into a narrow cap, whereas in 2048
	dimensions vectors of near-identical norm can still be near-orthogonal. The cap axis is the
	honest variable; where the *real* activations sit on it cannot be read off the committed
	CSVs and needs activations_layer30.npy.

	Usage:
		.venv/bin/python experiment2/viz/noneuclidean_power.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from experiment2.lib.alexandrov import cell_assignment, solve_weights_from_cost
from experiment2.lib.exp1_stats import (
	CLASS_NAMES, calibrated_cloud, covariance_similarity, load_reported_run, norm_stats,
)
from experiment2.lib.spherical import (
	from_tangent, karcher_mean, normalize, power_scores, tangent_basis, tangent_distortion,
	to_tangent,
)

CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
PALE = ["#dce9f8", "#fbe1d6", "#d5efe6"]
LINE_COLORS = {"chordal": "#e34948", "geodesic": "#4a3aa7"}
MUTED = "#52514e"
GEOMETRIES = ["chordal", "geodesic", "tangent"]
SEED = 0


# Twelve sub-sites per class, pushed back onto the sphere.
def fit_sites(points, labels, per_class):
	sites, owners, shape = [], [], []
	for index in range(len(CLASS_NAMES)):
		member = points[labels == index]
		# A class can hold fewer training points than the requested sub-site count; k-means
		# raises rather than degrading, so clamp the way discriminants.py already does.
		clusters = min(per_class, len(member))
		if clusters < 1:
			continue
		assignment = KMeans(n_clusters=clusters, n_init=10, random_state=SEED).fit_predict(member)
		for cluster in range(clusters):
			rows = member[assignment == cluster]
			if len(rows) == 0:
				continue
			sites.append(rows.mean(axis=0))
			owners.append(index)
			shape.append(len(rows))
	shape = np.array(shape, dtype=float)
	return normalize(np.array(sites)), np.array(owners), shape / shape.sum()


# One held-out configuration: points on the sphere, sites, owners, and class-prior targets.
def build_split(separation, layer, model, n_per_class, radial_share, per_class):
	ambient, labels = calibrated_cloud(
		layer, model, n_per_class=n_per_class, dims=3,
		radial_share=radial_share, separation=separation, seed=SEED,
	)
	points = normalize(ambient)
	rng = np.random.default_rng(SEED)
	order = rng.permutation(len(points))
	half = len(points) // 2
	train, test = order[:half], order[half:]
	sites, owners, shape = fit_sites(points[train], labels[train], per_class)

	prior = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
	targets = shape.astype(float).copy()
	for index in range(len(CLASS_NAMES)):
		owned = owners == index
		targets[owned] = targets[owned] / targets[owned].sum() * prior[index]
	return points[test], labels[test], sites, owners, targets


# Solve each geometry's weights and report the class each held-out point lands in.
def solve_all(points, sites, owners, targets):
	base = karcher_mean(points)
	basis = tangent_basis(base)
	masses = np.full(len(points), 1.0 / len(points))
	zero = np.zeros(len(sites))
	weights, assigned_class, plain_class = {}, {}, {}
	for geometry in GEOMETRIES:
		cost = power_scores(points, sites, zero, geometry, base=base, basis=basis)
		plain_sites, _ = cell_assignment(cost, zero, masses)
		solved, info = solve_weights_from_cost(cost, targets, masses)
		weights[geometry] = solved
		assigned_class[geometry] = owners[info["assigned"]]
		plain_class[geometry] = owners[plain_sites]
	return base, basis, weights, assigned_class, plain_class


# Fraction of the occupied cap that each geometry assigns differently from the tangent chart.
# Sampling the cap uniformly measures boundary displacement itself, rather than how many of a
# finite point set happen to sit near a boundary -- with 750 held-out points, one point is
# 0.13%, so the point-based version is mostly quantisation noise.
def cap_disagreement(sites, owners, weights, base, basis, radius, n_samples=40000, seed=0):
	rng = np.random.default_rng(seed)
	radii = radius * np.sqrt(rng.random(n_samples))
	angles = rng.uniform(0.0, 2.0 * np.pi, n_samples)
	coords = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
	sphere = from_tangent(coords, base, basis)

	def owning(geometry):
		scores = power_scores(sphere, sites, weights[geometry], geometry, base=base, basis=basis)
		return owners[np.argmin(scores, axis=1)]

	reference = owning("tangent")
	return {geometry: float(np.mean(owning(geometry) != reference))
	        for geometry in ("chordal", "geodesic")}


# Half-angle of the cap the data occupies, measured from the Karcher mean.
def cap_half_angle(points, base):
	return float(np.arccos(np.clip(points @ base, -1.0, 1.0)).max())


# Bisect the separation until a scalar readout hits its target (both are monotone in separation).
def bisect(readout, target, low=0.005, high=1.45, iterations=22):
	for _ in range(iterations):
		middle = 0.5 * (low + high)
		low, high = (low, middle) if readout(middle) > target else (middle, high)
	return 0.5 * (low + high)


# Axes of the shared tangent-chart canvas.
def canvas(coords, resolution):
	pad = 0.10 * max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
	x_axis = np.linspace(coords[:, 0].min() - pad, coords[:, 0].max() + pad, resolution)
	y_axis = np.linspace(coords[:, 1].min() - pad, coords[:, 1].max() + pad, resolution)
	return x_axis, y_axis


# Site and class index for every canvas pixel under one geometry.
def geometry_grid(x_axis, y_axis, sites, owners, weights, geometry, base, basis):
	site_grid = np.empty((len(y_axis), len(x_axis)), dtype=np.int16)
	for start in range(0, len(y_axis), 48):
		stop = min(start + 48, len(y_axis))
		mesh_x, mesh_y = np.meshgrid(x_axis, y_axis[start:stop])
		nodes = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
		scores = power_scores(from_tangent(nodes, base, basis), sites, weights, geometry,
		                      base=base, basis=basis)
		site_grid[start:stop] = np.argmin(scores, axis=1).reshape(stop - start, len(x_axis))
	return site_grid, owners[site_grid]


# Boolean mask of cells whose right or lower neighbour carries a different index.
def edge_mask(field):
	edges = np.zeros(field.shape, dtype=bool)
	edges[:, :-1] |= field[:, :-1] != field[:, 1:]
	edges[:-1, :] |= field[:-1, :] != field[1:, :]
	return edges


# Thicken a mask by one pixel, so class boundaries read heavier than sub-cell seams.
def thicken(mask):
	grown = mask.copy()
	grown[:, 1:] |= mask[:, :-1]
	grown[1:, :] |= mask[:-1, :]
	return grown


# Strip an axis of ticks and give it a recessive frame.
def bare(axis):
	axis.set_xticks([])
	axis.set_yticks([])
	for spine in axis.spines.values():
		spine.set_edgecolor("#bbbbbb")


# Draw one geometry's diagram in the shared tangent chart.
def draw_diagram(axis, coords, labels, site_coords, x_axis, y_axis, site_grid, class_grid, title):
	extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
	axis.imshow(class_grid, origin="lower", extent=extent, aspect="equal",
	            cmap=ListedColormap(PALE), vmin=0, vmax=2, interpolation="nearest")
	seams = edge_mask(site_grid) & ~edge_mask(class_grid)
	overlay = np.zeros(site_grid.shape + (4,))
	overlay[seams] = (0.45, 0.45, 0.45, 0.5)
	axis.imshow(overlay, origin="lower", extent=extent, aspect="equal", interpolation="nearest")
	overlay = np.zeros(site_grid.shape + (4,))
	overlay[thicken(edge_mask(class_grid))] = (0.04, 0.04, 0.04, 1.0)
	axis.imshow(overlay, origin="lower", extent=extent, aspect="equal", interpolation="nearest")
	for index, name in enumerate(CLASS_NAMES):
		member = coords[labels == index]
		axis.scatter(member[:, 0], member[:, 1], s=3.2, c=CLASS_COLORS[index],
		             linewidths=0, alpha=0.72, label=name, zorder=3)
	axis.scatter(site_coords[:, 0], site_coords[:, 1], s=14, c="black",
	             edgecolors="white", linewidths=0.6, zorder=4)
	axis.set_title(title, fontsize=10, linespacing=1.5)
	bare(axis)


# Highlight the pixels where a geometry parts company with the tangent chart.
def draw_disagreement(axis, coords, x_axis, y_axis, class_grids, share):
	extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
	axis.imshow(class_grids["tangent"], origin="lower", extent=extent, aspect="equal",
	            cmap=ListedColormap([shade + "55" for shade in PALE]), vmin=0, vmax=2,
	            interpolation="nearest")
	for geometry in ("chordal", "geodesic"):
		differs = class_grids[geometry] != class_grids["tangent"]
		overlay = np.zeros(differs.shape + (4,))
		rgb = matplotlib.colors.to_rgb(LINE_COLORS[geometry])
		overlay[thicken(differs)] = rgb + (0.9,)
		axis.imshow(overlay, origin="lower", extent=extent, aspect="equal",
		            interpolation="nearest")
	axis.scatter(coords[:, 0], coords[:, 1], s=2.0, c="#9a9a9a", linewidths=0, alpha=0.45, zorder=3)
	axis.set_title(f"Disagreement with the tangent chart\n"
	               f"chordal {share['chordal'] * 100:.1f}%, geodesic {share['geodesic'] * 100:.1f}%"
	               f" of canvas", fontsize=10, linespacing=1.5)
	bare(axis)


# Points reassigned relative to the tangent chart, as the data cap widens.
def draw_sweep(axis, sweep, exp1_cap_deg, exp1_share):
	angles = [row["cap_deg"] for row in sweep]
	for geometry in ("chordal", "geodesic"):
		axis.plot(angles, [row[geometry] * 100 for row in sweep], linewidth=2.0,
		          color=LINE_COLORS[geometry], label=f"{geometry} vs tangent")
	axis.axvline(exp1_cap_deg, color="#0b0b0b", linestyle=":", linewidth=1.6)
	axis.annotate(
		f"Experiment 1\ncalibrated cap {exp1_cap_deg:.0f}°\n"
		f"({max(exp1_share.values()) * 100:.1f}% of area)",
		xy=(exp1_cap_deg, axis.get_ylim()[1]), xytext=(6, -6),
		textcoords="offset points", va="top", fontsize=8.5, color="#0b0b0b",
	)
	axis.set_xlabel("cap half-angle spanned by the classes  (degrees)", fontsize=9)
	axis.set_ylabel("cap area assigned differently  (%)", fontsize=9)
	axis.set_title("Boundary displacement vs angular extent\nof the class distribution",
	               fontsize=10, linespacing=1.5)
	axis.legend(fontsize=8.5, framealpha=0.92)
	axis.grid(alpha=0.25, linewidth=0.6)
	axis.tick_params(labelsize=8)
	for spine in ("top", "right"):
		axis.spines[spine].set_visible(False)


# Calibrate, sweep the cap width, and write the figure.
def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", type=str, default=None,
	                    help="Experiment 1 model to read; defaults to the only collected run.")
	parser.add_argument("--n-per-class", type=int, default=500)
	parser.add_argument("--radial-share", type=float, default=0.35)
	parser.add_argument("--display-cap", type=float, default=72.0,
	                    help="Cap half-angle (deg) at which the diagram panels are drawn.")
	parser.add_argument("--sweep-points", type=int, default=16)
	parser.add_argument("--grid", type=int, default=440)
	parser.add_argument("--out", type=str, default="experiment2/figures/noneuclidean_power.png")
	args = parser.parse_args()

	run = load_reported_run(args.model)
	per_class = run["n_sites"] // len(CLASS_NAMES)
	stats = norm_stats(run["layer"], args.model)
	similarity = covariance_similarity(run["layer"], args.model)
	shared = (run["layer"], args.model, args.n_per_class, args.radial_share, per_class)

	# One configuration per separation, memoised so the bisections stay cheap.
	cache = {}

	def configure(separation):
		if separation not in cache:
			bundle = build_split(separation, *shared)
			solved = solve_all(bundle[0], bundle[2], bundle[3], bundle[4])
			cache[separation] = (bundle, solved)
		return cache[separation]

	def accuracy_at(separation):
		(points, labels, _, owners, _), (_, _, _, _, plain) = configure(separation)
		return float(np.mean(plain["chordal"] == labels))

	def cap_at(separation):
		(points, _, _, _, _), (base, _, _, _, _) = configure(separation)
		return np.degrees(cap_half_angle(points, base))

	exp1_separation = bisect(accuracy_at, run["power_accuracy"])
	(points, labels, sites, owners, targets), (base, basis, _, assigned, plain) = configure(
		exp1_separation
	)
	exp1_cap = cap_at(exp1_separation)
	exp1_coords = to_tangent(points, base, basis)
	exp1_share = cap_disagreement(
		sites, owners, cache[exp1_separation][1][2], base, basis,
		float(np.linalg.norm(exp1_coords, axis=1).max()),
	)
	distortion = tangent_distortion(points, base, basis)

	print(f"layer {run['layer']}  |  separation {exp1_separation:.4f} -> plain accuracy "
	      f"{accuracy_at(exp1_separation):.4f} (experiment 1 reported {run['power_accuracy']:.4f})")
	print(f"experiment 1 calibrated cap half-angle {exp1_cap:.1f} deg; tangent distortion "
	      f"mean {distortion['mean'] * 100:.2f}%, p95 {distortion['p95'] * 100:.2f}%")
	print(f"cap area differing from tangent: chordal {exp1_share['chordal'] * 100:.2f}%, "
	      f"geodesic {exp1_share['geodesic'] * 100:.2f}%")

	# Sweep the cap width and record how far each geometry drifts from the tangent chart.
	sweep = []
	print(f"\n{'cap deg':>9}{'chordal %':>12}{'geodesic %':>13}{'accuracy':>11}")
	for separation in np.linspace(0.02, 1.45, args.sweep_points):
		(pts, lab, s_sites, s_owners, _), (s_base, s_basis, s_weights, _, plain_at) = configure(
			float(separation)
		)
		radius = float(np.linalg.norm(to_tangent(pts, s_base, s_basis), axis=1).max())
		row = {"cap_deg": cap_at(float(separation))}
		row.update(cap_disagreement(s_sites, s_owners, s_weights, s_base, s_basis, radius))
		sweep.append(row)
		print(f"{row['cap_deg']:>9.1f}{row['chordal'] * 100:>12.2f}{row['geodesic'] * 100:>13.2f}"
		      f"{np.mean(plain_at['chordal'] == lab):>11.4f}")
	sweep.sort(key=lambda row: row["cap_deg"])

	# Draw the diagram panels at a cap wide enough for the geometries to be told apart.
	display_separation = bisect(cap_at, args.display_cap)
	(d_points, d_labels, d_sites, d_owners, _), (d_base, d_basis, d_weights, _, _) = configure(
		display_separation
	)
	d_coords = to_tangent(d_points, d_base, d_basis)
	d_site_coords = to_tangent(d_sites, d_base, d_basis)
	x_axis, y_axis = canvas(d_coords, args.grid)
	site_grids, class_grids = {}, {}
	for geometry in GEOMETRIES:
		site_grids[geometry], class_grids[geometry] = geometry_grid(
			x_axis, y_axis, d_sites, d_owners, d_weights[geometry], geometry, d_base, d_basis,
		)
	area_share = {
		geometry: float(np.mean(class_grids[geometry] != class_grids["tangent"]))
		for geometry in ("chordal", "geodesic")
	}
	print(f"\ndisplay cap {cap_at(display_separation):.1f} deg; canvas disagreement "
	      f"chordal {area_share['chordal'] * 100:.2f}%, geodesic {area_share['geodesic'] * 100:.2f}%")

	figure, axes = plt.subplots(1, 4, figsize=(17.2, 5.0))
	draw_diagram(axes[0], d_coords, d_labels, d_site_coords, x_axis, y_axis,
	             site_grids["geodesic"], class_grids["geodesic"],
	             "Geodesic  $d_g(x,y)^2-w$\nintrinsic, non-Euclidean")
	draw_diagram(axes[1], d_coords, d_labels, d_site_coords, x_axis, y_axis,
	             site_grids["tangent"], class_grids["tangent"],
	             "Tangent chart  $\\|u-v\\|^2-w$\nlog map at Karcher mean")
	draw_disagreement(axes[2], d_coords, x_axis, y_axis, class_grids, area_share)
	draw_sweep(axes[3], sweep, exp1_cap, exp1_share)
	axes[0].legend(loc="lower right", fontsize=8, framealpha=0.92, markerscale=2.2)

	figure.suptitle(
		f"Coordinate geometry and boundary placement  —  {run['model']}, layer {run['layer']}, "
		f"{len(sites)} sites, solved Alexandrov weights"
		f"\nDiagram panels drawn at a {cap_at(display_separation):.0f}° cap, where the arms are "
		f"visually separable; the sweep panel reports all cap widths",
		fontsize=11, y=0.995, linespacing=1.6,
	)
	norms = "  ".join(f"{name[:4]} {stats[name]['mean_norm']:.0f}±{stats[name]['std_norm']:.1f}"
	                  for name in CLASS_NAMES)
	figure.text(
		0.5, 0.052,
		f"Synthetic 3D stand-in calibrated to Experiment 1 layer-30 norms ({norms}) and to its "
		f"reported 36-site accuracy {run['power_accuracy']:.3f}, which fixes the cap at "
		f"{exp1_cap:.0f}°.",
		ha="center", fontsize=8, color=MUTED,
	)
	figure.text(
		0.5, 0.020,
		f"In 2048 dimensions that cap need not be narrow, so read the sweep axis rather than the "
		f"single marked point.    Measured covariance similarity: raw {similarity['raw']:.3f}, "
		f"PNS {similarity['pns']:.3f}.",
		ha="center", fontsize=8, color=MUTED,
	)
	figure.tight_layout(rect=[0, 0.075, 1, 0.935])

	destination = REPO_ROOT / args.out
	destination.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(destination, dpi=190)
	print(f"wrote {destination}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
