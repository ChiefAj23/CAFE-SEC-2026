# Surgery equivalence and metadata consumption

## Why this experiment exists

Topological model surgery replaces an opaque `C3K2` block with a traceable twin built from native
PyTorch primitives, realigns the convolution namespace, and migrates the framework's routing
metadata (`f`, `i`, `type`, `np`, `index`) onto the replacement. The correctness of that procedure
is currently established *indirectly* — the pruned models train and evaluate sensibly, so the
surgery is assumed to have preserved behavior.

Indirect evidence is weak here, for a specific reason. The twin's forward implementation is written
from scratch, so its internal execution order can differ from the original's even when the two are
mathematically equivalent on paper. If the reconstruction were subtly miswired, or if the migrated
metadata were not actually consumed, the resulting model would still run, still train, and still
produce plausible numbers — while silently computing something other than the original network.
Every downstream pruning and quantization result would inherit that error, and nothing in the
existing evaluation would surface it.

The claim also contains an assumption worth isolating: that copying the routing metadata is what
maintains correct forward-pass indexing. That assumption is only meaningful if the framework reads
those fields during inference. If a field is never read, copying it is inert, and citing it as part
of the correctness argument is unfounded.

Two separate checks follow, and the first does not depend on the surgery implementation at all.

## Check 1 — which metadata fields does the forward pass actually read?

`verify_metadata_consumption.py` answers this empirically rather than by inspection.

Because the metadata live in each module's instance `__dict__`, they cannot be intercepted with
`__getattr__`. Instead each value is replaced with a subclass of its own type — `int`, `str`, or
`list` — whose dunder methods increment a read counter. The instrumented model then runs a real
forward pass, and the counters report which fields were touched. A static scan of the framework's
forward implementations is reported alongside, as corroboration.

```bash
python experiments/surgery_equivalence/verify_metadata_consumption.py \
    --weights weights/P0/classifier.pt --task classify --imgsz 224
```

### Result (Ultralytics 8.4.14, classifier, one forward pass over 10 modules)

| Field | Defined on | Reads during forward | Verdict |
|---|---|---|---|
| `f` | 10/10 modules | **10** | **consumed — load-bearing** |
| `i` | 10/10 modules | **10** | **consumed — load-bearing** |
| `type` | 10/10 modules | 0 | inert during inference |
| `np` | 10/10 modules | 0 | inert during inference |
| `index` | **0/10 — attribute does not exist** | 0 | inert |

Only two of the five fields participate in inference, and they do so on every module of every
forward pass:

- **`f` controls input routing.** `tasks.py:177-178`: a module with `f != -1` takes its input from
  previously saved activations, either a single tensor `y[m.f]` or a list gathered across several
  indices. An incorrect `f` rewires the network.
- **`i` controls output retention.** `tasks.py:182`: `y.append(x if m.i in self.save else None)`
  decides which activations are kept for later concatenation and skip connections. An incorrect `i`
  starves a downstream consumer of its input.

`type` and `np` appear only in feature visualization and the profiling log; `np` is a parameter
count set during model construction. `index` is not an attribute of any module in this version of
the framework.

This is a useful negative result. The metadata migration is *not* correct because every field was
copied; it is correct because `f` and `i` were, and those two are the ones the execution path
consumes. That is the mechanism a port to another framework would need to reproduce, and it is
what the paper states in Sec. IV-E.

It is also a reassuring result about failure modes. Both consumed fields govern *structural*
wiring, so an error in either produces a shape mismatch or a `None` where a tensor is expected —
loud failures at the first forward pass, not silent numerical drift.

## Check 2 — output equivalence before and after surgery

**Status: pending.** Requires the surgery implementation (the `C3K2_v2` module, the
deep-copy/replace routine, and the weight-transfer step), which is not part of this repository.

The intended test: run the same inputs through the trained network and its post-surgery twin,
*before any pruning*, and report the divergence between them — maximum and mean absolute difference
on the output logits, plus the same at intermediate tap points around each replaced block. If the
surgery is lossless the two should agree to floating-point tolerance; any larger divergence
localizes a wiring error to a specific block. Because Check 1 establishes that `f` and `i` are the
fields the forward pass consumes, an equivalence result also validates that their migration was
performed correctly.

Running this before pruning is what makes it diagnostic. Pruning changes the network's outputs by
design, so a comparison made afterwards cannot separate an intended change from a surgical error.

## How this helps the research

The surgery is the part of this work that generalizes. The compression results are specific to one
architecture on two devices, but "here is how to make an attention-augmented, routing-opaque
backbone tractable for dependency-graph pruning" is a transferable procedure — and a transferable
procedure needs its correctness demonstrated at the structural level, not inferred from downstream
accuracy.

Check 1 already sharpens the claim: it identifies precisely which framework state the procedure
must preserve, replacing a list of five field names with a mechanism. Anyone adapting this surgery
to another framework now knows what to look for — the fields that determine input routing and
output retention — rather than copying an arbitrary list of attribute names that happens to work
for one library version.

Check 2 closes the remaining gap by verifying the reconstruction numerically, at the point in the
pipeline where a fault would still be attributable.
