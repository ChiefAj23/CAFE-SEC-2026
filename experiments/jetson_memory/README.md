# Measured runtime memory and power on the Jetson Orin Nano

## Why this experiment exists

The deployment argument prices classifier footprint as the single largest term in the ranking, and
treats a 180.6 MB classifier as disqualifying against a 7.4 GB shared envelope. Every part of that
argument is made from **serialized model size** — the size of a file on disk.

Serialized size is not the quantity that constrains deployment. A weight file does not include the
CUDA context, the execution context, the activation workspace, or the runtime's own allocations,
and on a unified-memory device every GPU allocation comes out of the same pool as the operating
system's. Serialized size could therefore understate the true cost, overstate it, or misrank models
against each other, and nothing in the evaluation establishes which. A metric that assigns 40% of
its weight to a proxy ought to check the proxy.

## What it does

Two scripts, both sampling `tegrastats` at 100 ms intervals around the workload and subtracting an
idle baseline measured in the same session.

**`measure_peak_memory.py`** — the deployment path. Builds a TensorRT FP16 engine on-device with
`trtexec`, then runs a fixed inference workload, recording peak RAM against the device total, delta
over idle, peak power, and engine size. Requires no PyTorch on the board, only the TensorRT that
ships with JetPack, which also keeps the measurement closer to a real deployment than a
framework-mediated one would be. The builder workspace is capped at 1 GB so that a failure means
something real rather than an unbounded allocation.

**`measure_ort_memory.py`** — a same-runtime comparison. The largest external classifier cannot be
compiled to a TensorRT engine on this board at all, so covering it requires a second runtime; and
comparing a TensorRT engine against an ONNX Runtime session would confound runtime with model.
This script therefore runs *every* classifier under one ONNX Runtime configuration, making the
models comparable to each other.

```bash
python3 measure_peak_memory.py     # TensorRT path, all models
python3 measure_ort_memory.py      # shared-runtime classifier comparison
```

Outputs are written to `jetson_memory.json` and `jetson_ort_memory.json`.

## What it establishes

Board: JetPack R36.3.0, TensorRT 8.6.2, `NV Power Mode: 15W`, 7620 MB total, idle 3637 MB at 5.2 W.

### Runtime memory vastly exceeds serialized size

| Model | Serialized | Engine | Peak RAM | Δ over idle | × serialized | Peak power |
|---|---|---|---|---|---|---|
| det_P0 | 10.5 MB | 9.66 MB | 4071 MB | 434 MB | **41×** | 9.90 W |
| det_P10 | 9.6 MB | 9.41 MB | 4148 MB | 511 MB | **53×** | 9.32 W |
| cls_P0 | 6.8 MB | 5.71 MB | 4028 MB | 391 MB | **57×** | 7.23 W |
| cls_P10 | 6.1 MB | 5.54 MB | 4010 MB | 373 MB | **61×** | 7.19 W |

A 6.8 MB classifier occupies 391 MB. The fixed cost of GPU inference — CUDA context, execution
context, workspace — is an order of magnitude larger than a nano-scale model itself, so at this
scale **the runtime, not the weights, dominates memory pressure**.

### Under one shared runtime, the size ordering survives

| Model | Serialized | Peak RAM Δ | × serialized | Mean latency |
|---|---|---|---|---|
| cls_P0 | 6.8 MB | 24 MB | 3.5× | 14.1 ms |
| cls_P10 | 6.1 MB | 24 MB | 3.9× | 12.9 ms |
| POSTERv2 | 180.7 MB | 239 MB | 1.3× | 239.6 ms |

Ten times the memory and seventeen times the latency. The multiplier over serialized size is not
constant — it is 3.5× for a small model and 1.3× for a large one, because fixed overhead dominates
the former and weights dominate the latter — so serialized size is a poor predictor of absolute
cost while still ranking models correctly by it.

### The largest external classifier cannot be compiled on the target device

`trtexec` fails on POSTERv2 after 9.3 s with an ONNX **parse** error: the opset-20 export contains
an operator TensorRT 8.6.2 cannot import, and no fallback plugin is registered. This is an
operator-support failure and explicitly **not** an out-of-memory condition. The practical
consequence is that the model is restricted to ONNX Runtime on this board, where it runs 17× slower
than the compiled alternative.

This also corrects a claim in the evaluation setup: opset 20 is *not* supported across the full
deployment stack on the Orin Nano. ONNX Runtime accepts it; TensorRT 8.6.2 does not.

### Power

Peak draw across all workloads is 9.90 W against a 5.2 W idle, within the board's configured 15 W
mode.

## How this helps the research

It replaces a proxy with a measurement, and the measurement changes what the argument rests on.

The original claim — that a large classifier is disqualifying — was inferred from file size. It now
rests on measured runtime memory, measured latency under a shared runtime, and a compilation
failure on the target device. That is a materially stronger basis, and it is the basis a
deployment-focused evaluation should have used from the start.

The more interesting finding is the 41–61× gap between file size and runtime cost. It means the
memory envelope binds *harder* than the paper claimed, not more loosely — and it exposes a limit of
the ranking metric itself, which prices a term it never measured. It also reframes what compression
buys at this scale: halving a 6 MB model saves 3 MB of weights against a ~380 MB fixed runtime cost,
so footprint reduction matters for whether a *pipeline of several models plus the OS* fits the
envelope, not because any single nano model is itself large. That is a more precise and more
defensible statement of why compression is necessary here.

Finally, the operator-support failure is a deployability constraint that no size- or accuracy-based
metric would ever surface. A model that cannot be compiled for the target runtime is undeployable
regardless of how it scores, which argues for treating runtime compatibility as a hard gate
alongside the memory and latency envelope.

### Scope

The TensorRT and ONNX Runtime tables are not comparable to each other: the ORT sessions here use the
CPU execution provider and allocate no CUDA context, which is why their deltas are far smaller.
Comparisons are valid within each table. Absolute values include whatever else the board was running
— the idle baseline is measured in the same session so the deltas remain attributable.
