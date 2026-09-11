# Precision / compilation ablation

## Why this experiment exists

The pipeline's headline speedup compares a P0-FP32 configuration executed in native PyTorch against
a P10-FP16 configuration executed as a TensorRT engine. That comparison moves three variables at
once — pruning ratio, numeric precision, and runtime — so it cannot attribute the resulting
acceleration to any one of them.

The paper already contains the clue that this matters. Table IV reports P0-FP32 at 9.04 ms and
P10-FP32 at 9.02 ms: structured pruning at fixed precision and runtime changes latency by 0.02 ms.
If pruning does nothing at FP32, it is implausible that it is responsible for a 3.3× improvement
when precision and runtime change simultaneously. The paper states this honestly but does not
measure the decomposition, which leaves the central deployment claim resting on a confounded
comparison.

The missing arm is **P0-FP16**: the *unpruned* model compiled to a TensorRT FP16 engine. Adding it
factors the speedup into two independent effects that can be attributed separately:

```
P0-FP32  →  P0-FP16     compilation + precision, architecture held fixed
P0-FP16  →  P10-FP16    pruning, runtime and precision held fixed
```

## What it does

`ablate_precision.py` measures twelve arms — {P0, P10, P15} × {FP32 PyTorch, FP16 TensorRT} for both
the detector and the classifier — under conditions designed so the ratios are trustworthy even when
absolute latencies are hardware-specific:

- all arms in a **single process on a single GPU**, so driver state, clocks, and thermal conditions
  are shared;
- identical 20-image warmup before every measured run;
- identical image ordering across arms;
- latency read from Ultralytics' own `speed` dictionary for **both** the PyTorch and engine paths.
  Hand-timing PyTorch while reading framework timings for engines would bias precisely the
  comparison the experiment exists to make.

Classifier arms are evaluated on the full 3589-image FER2013 held-out split, so accuracy is
reported alongside latency; detector arms run the 20-scene stress set for a configurable number of
passes (default 10) to stabilize the estimate from a small image pool. Engines are compiled once
and cached.

```bash
python experiments/precision_ablation/ablate_precision.py            # full
python experiments/precision_ablation/ablate_precision.py --limit 200  # quick check
```

Outputs land in `results/validation/`: `precision_ablation.csv` (one row per arm) and
`precision_ablation.json` (arms, decomposition, and the recorded environment).

## What it establishes

Measured on an RTX PRO 5000 Blackwell with TensorRT 10.15.1.29 and Ultralytics 8.4.14:

| Stage | Headline P0-FP32 → P10-FP16 | = Compilation + FP16 | × Pruning |
|---|---|---|---|
| Classifier | 3.394× | **3.418×** | **0.993×** |
| Detector | 3.882× | **3.804×** | **1.020×** |

**Pruning contributes no latency**, and for the classifier is fractionally negative. Compilation and
FP16 arithmetic account for essentially the entire speedup — 3.418× of 3.394× and 3.804× of 3.882×.

Two secondary results fall out:

**FP16 quantization is accuracy-neutral.** Top-1 goes 72.72% → 72.78% at P0 and 70.27% → 70.27% at
P10. The pipeline's accuracy cost is therefore attributable to pruning alone (72.72% → 70.27%,
−2.45 points at P10), separated from precision for the first time.

**Pruning's contribution is footprint, not speed.** The pruned checkpoints are smaller by
construction, and that reduction is what fits the pipeline inside a constrained shared-memory
envelope — but it does not translate into latency at nano scale, where layer channel counts of 8–64
already fail to saturate the GPU's parallel execution units.

## How this helps the research

It replaces a confounded claim with an attributed one, and the attributed version is both more
useful to a practitioner and more defensible.

A reader deciding how to deploy this pipeline needs to know *which* step to spend effort on. The
confounded 3.3× implies that compression is the lever. The decomposition shows it is not: on a
modern GPU the compiler is the lever, and it is available without any architectural change at all.
Structured pruning earns its place by reducing footprint — which is what determines whether the
model fits the memory budget, a hard deployment constraint rather than a performance optimization.

That reframing does not diminish topological model surgery; it locates it correctly. Surgery is not
a latency mechanism and was never measured as one. It is what makes attention-augmented YOLOv12n
*prunable at all* — standard dependency-graph tracers fail on the C3K2 routing and A2C2f attention
constraints, so without surgery the footprint reduction is unavailable at any latency. The honest
statement of the contribution is that the two effects compose along different axes: TensorRT and
FP16 deliver the latency, surgery-enabled pruning delivers the footprint, and a deployment needs
both.

It also explains the paper's own counterintuitive INT8 result. If arithmetic width were the binding
constraint, INT8 would beat FP16; instead INT8 is slower, because per-layer dequantization overhead
exceeds the arithmetic savings on small tensors. Both observations point at the same underlying
fact — at nano scale, latency is dominated by kernel scheduling and memory traffic rather than by
arithmetic volume, so compression strategies that reduce arithmetic do not reduce time.

### Scope

Absolute latencies are specific to the GPU used and will not match measurements from other
hardware; the ratios are the claim, and they rest on all arms sharing one device and one process.
The decomposition is component-level. A pipeline-level P0-FP16 arm requires the end-to-end
configuration sweep, which is not part of this repository.
