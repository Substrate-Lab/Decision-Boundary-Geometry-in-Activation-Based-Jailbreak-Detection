# Layer-separation analysis — notes and retrospective (2026-08-23)

Model: Qwen2.5-3B-Instruct (4-bit). Data: balanced 500/500/500 (Refusal / Jailbreak /
Benign), stratified across SORRY-Bench, WildJailbreak, ToxicChat, Aegis 2.0. Layers
2–36 (every 2), last-token residual-stream activations.

## What we expected
A "violin" shape across depth: class separation small at the early and late layers and
**largest in the middle layers** — the proof-of-concept that middle layers hold the most
"decision geometry", which would justify putting the method's sub-centroids there.

## What we actually found
No middle hump. Class separation **rises with depth and peaks in the upper layers
(~24–30)**; the geometric middle (~18) is among the *weakest*. This held up under every
variation we tried:

| Candidate tested | Result |
| --- | --- |
| Behavioral labels (refuse/comply) | Monotonic rise to ~0.78, flat late. No hump. |
| Prompt labels (harmful/benign concept) | Peak ~0.886 at layer 26, mild taper to 0.85. Upper-layer peak, not middle. |
| Nearest-centroid → logistic probe | Lifts every layer's accuracy; same rising shape. |
| PCA dims 10 → 50 → 100 | Minor lift (mostly early layers); same shape. |
| Last-token → mean-pooled tokens | Same shape; slightly higher early layers; still no middle peak. |

Covariance mismatch (the paper's core quantity) is also a **late** phenomenon: raw
between-class covariance similarity stays ~0.9 through the middle and drops to ~0.67 at
layer 34 (most mismatch), i.e. the classes' spreads diverge most in the late layers.

## Why this "isn't good" — and why it partly is
It is not the result the POC wanted: the middle-layer story is not supported for this
model and measurement. But two things make it a *usable* result rather than a failure:
1. The negative is **robust** — four independent variations agree, so it is not an
   artifact of one metric or setting.
2. The covariance mismatch the power-diagram method exploits is **strongest exactly where
   it is most useful** (late layers), which is good for the method even though it breaks
   the middle-layer framing.

## Where the reasoning went wrong (the actual analysis)
1. **Conflating concept representation with decision read-out.** The "middle layers are
   richest" intuition comes from probing work on where *features/concepts* live. We
   measured, at the **last token**, quantities that track the model's *decision*
   (refuse/comply). A decoder-only model's decision necessarily sharpens toward the
   output, so measuring it and expecting a middle peak is a category mismatch.
2. **Behavioral labels bias the curve late.** Because the labels are defined by the
   model's own output, "how well layer L predicts the label" almost tautologically
   increases with depth. That alone converts any middle hump into a rising curve. Using
   ground-truth prompt labels (harmful/benign) is what recovered even a mild peak-and-taper.
3. **The violin shape was an assumed prior, not a derived prediction.** Nothing in the
   setup guarantees a symmetric middle bump; for last-token decision geometry a late peak
   is the expected outcome.
4. **Mean-pooling was the wrong fix and would hurt the method.** It slightly lifts early
   layers but does not create a middle peak — and it *destroys* the covariance mismatch
   (similarity stays 0.95–0.999 at every layer), erasing the exact signal the power
   diagram is built to exploit. Last-token is the correct choice for this project.

## What was NOT wrong
- The pipeline is correct (batched left-padded extraction, stratified sampling, balanced
  downsampling, PNS unwrapping) and was independently reviewed.
- The data is clean: balanced classes, even dataset representation, ~49% refusal rate.
- The measurement was thorough — we falsified the middle-layer idea four different ways
  rather than trusting one plot.

## Open caveats / what could still change the picture
- **One model only** (Qwen-3B, 36 layers). Larger models may place the decision band
  differently; "middle" may scale with depth.
- **"Middle" definition.** The concept probe does peak at ~24–26 (upper-middle) and taper
  after — if the claim is relaxed to "not the very last layers", there is a peak-and-decline.
- **Token position.** We tried last-token and mean-pool; a targeted position (e.g. the last
  content token before the chat-template suffix, or per-class steering directions) is still
  untested and is the only remaining lever on this axis.

## Takeaway for the paper / PI
Report the middle-layer POC as **not supported at fine resolution**: decision geometry and
covariance mismatch are **late-layer** (~24–34) phenomena in Qwen-3B, robust across label
choice, probe strength, dimensionality, and token pooling. Keep **last-token** activations
(mean-pooling erases the covariance mismatch). The covariance-mismatch result itself is
clean and is strongest where the method needs it.
