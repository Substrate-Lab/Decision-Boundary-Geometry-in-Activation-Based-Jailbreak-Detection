# Experiment 1 — Decision-Boundary Geometry in a 2D Toy Lab

## What this is

Experiment 1 is the foundational proof-of-concept for the whole paper. It is a
controlled, two-dimensional "toy laboratory" that deliberately strips away the
messiness of real Large Language Models (LLMs) so we can test the core
mathematical claim in a setting where we already know the exact right answer.

The claim is simple to state:

> When one class of points is tightly packed and another is widely scattered
> (a *covariance mismatch*), a standard linear probe draws its decision boundary
> in the wrong place. A **Power Diagram** puts the boundary back where it belongs.

Because this is a synthetic setup, we can compute the mathematically perfect
boundary and check each method against it. If the idea does not work here, there
is no point taking it to a real model. If it does work here, we have a solid
foundation to build on.

## Two levels of understanding

### Level 1 — the geometric sandbox

We create two groups of points in a 2D plane, each drawn from a Gaussian (bell-shaped)
distribution:

- **Class A (Refusal)** is centered at `mu_A = [-2, 0]` with tight spread,
  `Sigma_A = I` (the identity matrix — a small, round cluster).
- **Class B (Jailbreak)** is centered at `mu_B = [2, 0]` with spread scaled by a
  factor `k`, so `Sigma_B = k * I`.

We run two scenarios, which we call *strata*:

- **Low-Mismatch (`k = 1`)** — both classes have equal spread.
- **High-Mismatch (`k = 10`)** — Class B is ten times more diffuse than Class A.

When the two spreads match, a flat boundary placed at the midpoint between the
two centers separates them cleanly. But when Class B is far more spread out, that
same flat midpoint boundary slices straight through the sprawling tail of Class B
— labelling many genuine Class B points as if they were Class A.

### Level 2 — the reality this simulates

The toy setup is a stand-in for something concrete that happens inside a real
LLM's *residual stream* (its internal activation space):

- **Class A mirrors the "refusal attractor."** Safety fine-tuning tends to
  collapse many different harmful prompts into one stereotyped, dense,
  low-variance "refusal" direction. Refusals look alike, so they cluster tightly.
- **Class B mirrors the "compliance manifold."** A successful jailbreak makes the
  model do all sorts of different tasks — writing poetry, decoding Base64,
  producing Python, playing a role. These varied behaviours push activations out
  into a broad, high-variance region.

So a flat midpoint boundary — exactly what a linear probe learns — will tend to
misclassify the more unusual jailbreaks as safe. That is the failure mode we want
to catch and fix.

```
LOW-MISMATCH (k = 1): midpoint fence works
        |
   A A  |  B B
  A A A | B B B
   A A  |  B B
        |
   mu_A    mu_B
   fence sits at the midpoint; both clusters are equally tight


HIGH-MISMATCH (k = 10): midpoint fence cuts off diffuse Class B
              |
              |    B         B
   A A        |  B    B   B      B
  A A A       | B   B    B    B    B     <- Class B sprawls
   A A        |  B    B   B      B
              |    B         B
              |
   mu_A         mu_B
   fence still at the midpoint, but Class B's wide tail spills across it
   and gets wrongly labelled as Class A
```

## The five methods compared

- **Bayes Optimal (Ground Truth)** — computed directly from the known true
  parameters; the mathematically perfect boundary that nothing can beat.
- **QDA (Quadratic Discriminant Analysis)** — estimates the means and covariances
  from data to form a curved boundary; the gold-standard statistical benchmark.
- **Linear Probe / Logistic Regression** — fits a single flat line; ignores
  covariance entirely.
- **Plain Voronoi** — places the boundary at the strict midpoint between the two
  centroids using ordinary (Euclidean) distance; also ignores covariance.
- **Power Diagram (our method)** — uses the centroids plus a scalar weight per
  class, `w_c = trace(Sigma_hat_c)`, taken from each class's variance. The
  boundary is where the *power distances* are equal:
  `||x - mu_A||^2 - w_A = ||x - mu_B||^2 - w_B`.

## How to run

Run the three scripts in order:

1. `python step1_generate_data.py` — generate the synthetic 2D data for both
   strata; writes CSV files into `data/`.
2. `python step2_fit_boundaries.py` — fit all five methods and extract/plot their
   decision boundaries; writes boundary figures into `figures/`.
3. `python step3_evaluate.py` — score every method against the Bayes-optimal
   ground truth; writes a metrics table into `data/` and a summary figure into
   `figures/`.

Figures are produced only when the scripts are actually run.

## Metrics

- **Misclassification rate and ROC AUC** — measured on held-out test points not
  used for fitting.
- **Boundary Distance Error (RMS)** — sample points along the true Bayes boundary,
  then measure the perpendicular distance from each of those points to a method's
  boundary and report the root-mean-square.
- **Boundary Area Error** — the area of the 2D grid where a method's prediction
  disagrees with the Bayes-optimal prediction.

## What success looks like

| Method                    | High-Mismatch (k = 10) | Low-Mismatch (k = 1) |
| ------------------------- | ---------------------- | -------------------- |
| Linear Probe / Voronoi    | High error             | Low error            |
| QDA                       | Low error              | Low error            |
| Power Diagram (ours)      | Low error (≈ QDA)      | Low error (≈ QDA)    |

**Victory condition.** On the high-mismatch stratum (`k = 10`), the Power Diagram
must reach a Boundary Distance Error (RMS) far below that of the Linear Probe and
comparable to QDA — while keeping a clean, piecewise-linear boundary rather than a
fully curved one. Meeting this bar proves the mathematical foundation is sound and
justifies moving on to real LLM activations in Experiment 2.

## Folder layout

- `README.md` — this document.
- `step1_generate_data.py` — generates the synthetic data for both strata.
- `step2_fit_boundaries.py` — fits the five methods and extracts their boundaries.
- `step3_evaluate.py` — scores each method against the Bayes-optimal ground truth.
- `data/` — synthetic CSVs, the `metrics.csv` results table, and `params.json`
  recording the generating parameters.
- `figures/` — boundary and summary figures (produced on run).
