# Power-Diagram-Decision-Boundaries-for-Jailbreak-Detection
Power-Diagram Decision Boundaries for Jailbreak Detection - NeurIPS Interpretability for Discovery

Power-Diagram Decision Boundaries for Jailbreak Detection
Experimental Design Document

COPY PASTED FROM https://docs.google.com/document/d/1vaH21Lwiu2GTkCWdYvV40_pdSF88aGA7ssnBOq1Y3h0/edit?tab=t.0

1. The Problem
Deployed language models are guarded by refusal behavior: when a prompt requests something harmful, the model is supposed to decline. Jailbreaks defeat this — a rephrased, encoded, or role-played version of a harmful request slips past the refusal and the model complies. Detecting jailbreaks reliably, and knowing how confident that detection is, remains an open safety problem.
Most current detection approaches fall into two buckets, each with a gap:
Output/text classifiers read the prompt or the response and flag harmful content. They are brittle to novel phrasings and adversarial rewording — the whole point of a jailbreak is to look benign on the surface.
Activation probes (linear probes, logistic regression on hidden states) detect the internal signature of compliance-vs-refusal and are markedly more robust, because the model's internal state "knows" it is complying before the tokens appear. This is the right layer to work at. But a linear probe returns a bare score against a threshold: it draws a single flat separating hyperplane and gives no principled account of where the boundary should sit when the classes have very different internal spread, and no geometrically meaningful notion of margin — how far inside or outside the safe region a given prompt is.
The unaddressed gap: jailbreak and safe states are separable in activation space, but we lack a principled, geometry-aware decision boundary that (a) places the boundary correctly when classes have unequal covariance and (b) yields a calibrated, interpretable distance-to-boundary as a confidence signal.
2. The Solution (High-Level)
We model the activation space of an instruction-tuned LLM as a set of classes — jailbreak-compliance, refusal, and benign-compliance — each represented not by a point but by a weighted site: a centroid together with a weight derived from the class covariance (its spread in activation space). We then build a power diagram (Laguerre–Voronoi tessellation) over these weighted sites.
The power diagram is the key object. Unlike a plain Voronoi diagram, which splits territory at the midpoint between centroids and so mis-places the boundary whenever two classes have different spread, a power diagram uses power distance (squared Euclidean distance minus the site weight). A more diffuse class earns a larger cell; the boundary lands where the two classes are genuinely equiprobable given their spreads, not at the naive midpoint. The boundaries remain exact hyperplanes, so the partition is clean, convex, and — crucially — the signed power distance to the boundary is a well-defined, computable quantity for any prompt's activation.
That signed distance is our geometric margin: a principled confidence signal. A prompt deep inside the jailbreak cell is a high-confidence detection; one near the boundary is flagged as uncertain. This is precisely what a linear-probe threshold cannot give in a covariance-aware way.
The claim is deliberately bounded. We do not claim to beat linear probes on raw detection accuracy. We claim to match their detection performance while adding (1) a covariance-correct boundary that outperforms specifically on class pairs with mismatched spread, and (2) a calibrated geometric margin that a flat linear threshold does not provide. This is a "same accuracy, strictly more information, correct where the baseline is wrong" contribution.
3. Motivation — Why This Matters
Safety-relevant and legible. Jailbreak detection is a live deployment problem, and a calibrated detector (one that says "I'm 95% sure" vs "this is borderline") is far more useful operationally than a bare flag — borderline cases can be routed to heavier review.
The covariance point is real, not cosmetic. Compliance and refusal states genuinely differ in spread — refusals are stereotyped and tight; jailbroken compliance is heterogeneous (many attack styles). A boundary method that ignores this places the boundary wrong exactly where it matters. The power diagram is the principled fix, and its advantage is predicted in advance by the covariance mismatch, not fished for after the fact.
Interpretable geometry. The method yields an exact boundary and a signed margin, so every decision is explainable as "this prompt sits at power-distance d inside/outside the safe region." That interpretability is the thing prompting- and probe-based detectors lack.
4. Method
4.1 Activation collection. Hook a chosen layer (or a small set of layers) of an instruction-tuned LLM. For each prompt, capture the activation at the decision-relevant position (the point where the model commits to comply or refuse). This produces a labeled point cloud in the model's hidden dimension.
4.2 Class sites and weights. For each class c ∈ {jailbreak-compliance, refusal, benign}:
site centroid μ_c = mean activation of class c
weight w_c derived from the class covariance Σ_c (e.g. a scalar summary such as mean variance / generalized variance, so that a more diffuse class receives a larger weight and hence a larger cell). The exact weight functional is a design parameter swept in §5.
4.3 Power diagram construction. Build the Laguerre–Voronoi tessellation over the weighted sites. Assignment uses power distance π_c(x) = ‖x − μ_c‖² − w_c. A point x is assigned to the class minimizing π_c(x); the boundary between two classes is the hyperplane where their power distances are equal.
4.4 The geometric margin. For any activation x, define the signed margin as the difference between the two smallest power distances (distance to assigned cell vs. nearest competing cell). Large margin = confident; near-zero = borderline. This is the calibrated confidence signal and the paper's distinctive output.
4.5 Detection. A held-out prompt is classified by which power cell its activation falls into (jailbreak-compliance cell → flagged). Detection performance and the margin's calibration are both evaluated.
5. Experiments
The arc is separability → detection → margin value → robustness. Separability is the in-sample evidence; detection is the out-of-sample headline; margin and robustness are what make it more than a re-skinned probe.
Experiment 1 — Separability (in-sample). Show the classes occupy geometrically distinct regions and the power-diagram boundary separates them with a measurable margin on training data. Report the fraction of correctly-assigned points and the margin distribution per class. Success criterion: clean separation with margins that are systematically larger for unambiguous prompts. This is the foundation; if classes don't separate, nothing downstream holds.
Experiment 2 — Detection (out-of-sample, the headline). Hold out prompts (and, importantly, held-out jailbreak families — see robustness) and report detection AUC / accuracy / precision-recall for jailbreak-compliance vs. safe. This is where the "match the baseline" claim is tested.
Experiment 3 — The covariance win. Stratify results by class-pair covariance mismatch. The pre-registered prediction: the power diagram's advantage over plain Voronoi and over a flat linear boundary concentrates on the high-mismatch pairs (tight refusal vs. diffuse jailbreak). Show that where spreads are similar, power ≈ plain; where spreads differ, power places the boundary correctly and the baselines don't. This is the experiment that justifies the whole method choice.
Experiment 4 — Margin calibration. Show the geometric margin is a usable confidence signal: bin predictions by margin and show accuracy rises monotonically with margin (reliability curve); show that abstaining on low-margin cases (selective prediction) raises precision on the rest. This is the concrete operational value the linear-probe threshold can't match.
Experiment 5 — Robustness / generalization.
Cross-family: train on some jailbreak families, test on unseen ones (the realistic deployment condition). A detector that only catches seen attacks is not useful.
Cross-model: rerun the whole pipeline on a second and third instruction-tuned LLM; report consistency of the detection and margin results.
Control: on random-weight / shuffled-label versions, separability and detection should collapse to chance — confirming the signal reflects real learned structure, not a geometric artifact of the construction.
6. Benchmarks and Baselines
Datasets. Standard jailbreak / refusal corpora for the harmful side (e.g. AdvBench / HarmBench-style harmful requests with jailbreak wrappers) paired with matched benign requests and genuine refusals. Prompts constructed as matched pairs (same underlying request, harmful-wrapped vs. benign) so the contrast isolates the jailbreak signal rather than topic.
Baselines the paper must include (a detection paper is judged against these):
Linear probe / logistic regression on the same activations — the primary baseline. The claim is match on AUC.
Plain (unweighted) Voronoi on the same sites — isolates the value of the covariance weighting specifically.
Mahalanobis / GMM class-conditional classifier — a covariance-aware statistical baseline, to show the power diagram matches principled covariance methods while additionally giving an exact boundary and margin.
(Optional) an output-text jailbreak classifier — to demonstrate the activation-level approach's robustness advantage on adversarial rephrasings.
Metrics. Detection: AUC, accuracy, precision/recall, FPR at fixed TPR. Margin: reliability/calibration curve, selective-prediction accuracy-vs-coverage, ECE. Covariance win: performance delta vs. baselines as a function of class-pair covariance mismatch. Consistency: variance of all of the above across models and jailbreak families.
Pre-registered success criteria.
Detection AUC statistically indistinguishable from the linear probe (this is a "match" claim — state it as an equivalence, not a superiority).
Strict improvement over plain Voronoi and over the linear boundary on the high-covariance-mismatch stratum.
Monotone margin–accuracy reliability curve; selective prediction improves precision.
Full collapse to chance on controls.
7. Importance
If it works, this contributes a detection method with a principled geometric decision boundary and a calibrated confidence margin — matching the accuracy of the standard activation-probe baseline while (a) placing the boundary correctly under covariance mismatch, where the standard baseline is provably mis-placed, and (b) supplying an interpretable distance-to-boundary that supports selective prediction and human-in-the-loop routing. It reframes activation-level jailbreak detection from "a threshold on a probe score" to "an exact, covariance-aware geometric boundary with a measurable margin," and does so in a way that is model-agnostic and validated against controls.
The bounded framing is the strength: no causal claim, no universal claim, no dependence on any prior steering work. A separability result, an out-of-sample detection result that matches a strong baseline, a covariance-mismatch stratum where it wins, a calibrated margin, and a control that collapses. Every claim is measurable and defended by an experiment.
8. Scope Boundaries (what this paper does NOT claim)
No claim that the geometry causes or constitutes refusal/compliance — only that it is measurably separable and detectable.
No claim to beat linear probes on raw detection — the claim is match + geometric margin + covariance-correct boundary.
No cross-architecture universality claim beyond the models actually tested; broader generalization is stated as future work, not result.
No steering or intervention component.
9. Open Design Decisions to Lock Before Building
Weight functional w_c from Σ_c — scalar variance summary vs. generalized variance vs. log-det; sweep and pick by Experiment 3 behavior.
Layer choice — single decision layer vs. small multi-layer set; pick by Experiment 1 separability.
Decision-position capture — which token position counts as the comply/refuse commitment.
High-dimensional power diagram — computing exact tessellations in full hidden-dim is expensive; decide between operating in a reduced subspace (and justifying faithfulness) vs. computing power distances directly without materializing the full diagram (only the per-class power distances are needed for assignment and margin, which avoids explicit tessellation — likely the right move).

