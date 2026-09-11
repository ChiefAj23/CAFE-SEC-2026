"""Which routing-metadata fields does the forward pass actually read?

Topological model surgery replaces an opaque module with a traceable twin and
migrates the framework's routing metadata (`f`, `i`, `type`, `np`, `index`)
onto the replacement. The correctness argument for that migration rests on
those fields being consumed during inference -- if a field is never read, then
copying it is inert, and the claim that copying it preserves forward-pass
behavior is unfounded.

That is an empirical question about the framework, and it is answerable without
any reference to the surgery itself. This script answers it two ways:

  1. Statically, by locating each field's occurrences in the framework's
     forward implementations.
  2. Dynamically, by replacing each field's value with an instrumented object
     that records every read, then running a real forward pass and reporting
     which fields were actually touched.

The dynamic check is the load-bearing one: it observes consumption rather than
inferring it from source inspection.

Usage:
    python experiments/surgery_equivalence/verify_metadata_consumption.py \
        --weights weights/P0/classifier.pt --task classify
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

FIELDS = ("f", "i", "type", "np", "index")

reads: dict[str, int] = defaultdict(int)


def _tracer(field: str, base):
    """Build a value that behaves like `base` but records every read.

    The metadata live in each module's instance __dict__, so they cannot be
    intercepted with __getattr__. Instead each value is replaced by a subclass
    of its own type whose dunder methods -- the operations the forward pass
    performs on it -- increment a counter.
    """

    if isinstance(base, bool) or base is None:
        return base

    if isinstance(base, int):
        class TracedInt(int):
            def __eq__(self, other):
                reads[field] += 1
                return int(self) == other

            def __ne__(self, other):
                reads[field] += 1
                return int(self) != other

            def __index__(self):
                reads[field] += 1
                return int(self)

            def __hash__(self):
                # `m.i in self.save` hashes the value for set membership.
                reads[field] += 1
                return int.__hash__(self)

        return TracedInt(base)

    if isinstance(base, str):
        class TracedStr(str):
            def __eq__(self, other):
                reads[field] += 1
                return str(self) == other

            def __hash__(self):
                reads[field] += 1
                return str.__hash__(self)

        return TracedStr(base)

    if isinstance(base, list):
        class TracedList(list):
            def __iter__(self):
                reads[field] += 1
                return list.__iter__(self)

            def __eq__(self, other):
                reads[field] += 1
                return list(self) == other

            def __ne__(self, other):
                reads[field] += 1
                return list(self) != other

        return TracedList(base)

    return base


def static_scan() -> dict:
    """Locate each field in the framework's forward implementations."""
    import ultralytics.nn.tasks as tasks

    src_path = Path(inspect.getfile(tasks))
    src = src_path.read_text()
    lines = src.splitlines()

    # Line ranges of every _predict_once-style forward implementation.
    forward_spans = []
    for m in re.finditer(r"def (_predict_once|_predict_augment)\(", src):
        start = src[: m.start()].count("\n")
        forward_spans.append((start, start + 40))

    out = {}
    for field in FIELDS:
        pattern = re.compile(rf"\bm\.{field}\b")
        hits = [(n + 1, lines[n].strip())
                for n in range(len(lines)) if pattern.search(lines[n])]
        in_forward = [
            (n, text) for n, text in hits
            if any(lo < n <= hi for lo, hi in forward_spans)
            and "feature_visualization" not in text
            and "LOGGER" not in text
        ]
        out[field] = {
            "total_occurrences": len(hits),
            "in_forward_path": len(in_forward),
            "forward_sites": [f"{src_path.name}:{n}: {t}" for n, t in in_forward[:4]],
        }
    return out


def dynamic_check(weights: str, task: str, imgsz: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights, task=task)
    net = model.model
    net.eval()

    original = []
    for module in net.model:
        for field in FIELDS:
            if hasattr(module, field):
                base = getattr(module, field)
                original.append((module, field, base))
                setattr(module, field, _tracer(field, base))

    reads.clear()
    with torch.no_grad():
        net(torch.zeros(1, 3, imgsz, imgsz))
    observed = dict(reads)

    for module, field, base in original:
        setattr(module, field, base)

    present = {f: sum(1 for m in net.model if hasattr(m, f)) for f in FIELDS}
    return {"reads_during_forward": observed, "modules_defining_field": present}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--task", default="classify")
    ap.add_argument("--imgsz", type=int, default=224)
    args = ap.parse_args()

    static = static_scan()
    dynamic = dynamic_check(args.weights, args.task, args.imgsz)
    observed = dynamic["reads_during_forward"]

    print(f"{'field':8s} {'defined on':>11s} {'fwd sites':>10s} {'reads/forward':>14s}  verdict")
    verdicts = {}
    for field in FIELDS:
        n_mod = dynamic["modules_defining_field"][field]
        n_static = static[field]["in_forward_path"]
        n_read = observed.get(field, 0)
        consumed = n_read > 0
        verdicts[field] = "consumed" if consumed else "not consumed"
        print(f"{field:8s} {n_mod:11d} {n_static:10d} {n_read:14d}  "
              f"{'CONSUMED' if consumed else 'inert during inference'}")

    print("\nforward-path sites:")
    for field in FIELDS:
        for site in static[field]["forward_sites"]:
            print(f"  {field}: {site}")

    out = Path(__file__).resolve().parents[2] / "results" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata_consumption.json").write_text(json.dumps(
        {"weights": args.weights, "static": static, "dynamic": dynamic,
         "verdicts": verdicts}, indent=2))
    print(f"\nwrote {out / 'metadata_consumption.json'}")


if __name__ == "__main__":
    main()
