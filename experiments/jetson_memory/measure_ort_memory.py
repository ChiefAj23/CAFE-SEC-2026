"""Runtime memory of classifiers under a single shared runtime.

The TensorRT measurement in `measure_peak_memory.py` cannot cover the largest
external classifier: its opset-20 export contains an operator that TensorRT
8.6.2 on this board fails to import, so no engine can be built for it. That is
itself a deployability result, but it leaves the memory comparison incomplete
-- and comparing a TensorRT engine against an ONNX Runtime session would
confound runtime with model.

This script closes that gap by running every classifier under the *same*
runtime (ONNX Runtime, CPU execution provider) and sampling tegrastats around
each. The absolute numbers are therefore not comparable to the TensorRT figures
-- no CUDA context is allocated here -- but the models are directly comparable
to each other, which is the question at issue: does a 180.6 MB serialized
classifier impose proportionally more runtime memory pressure than a 6.8 MB
one, or does the fixed runtime overhead dominate?

Usage (on the Jetson):
    python3 measure_ort_memory.py
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path.home() / "cafe_memtest"
ONNX_DIR = ROOT / "onnx"
OUT = ROOT / "jetson_ort_memory.json"
TEGRASTATS = "/usr/bin/tegrastats"

SAMPLE_MS = 100
IDLE_SECONDS = 6
ITERATIONS = 40
WARMUP = 3

MODELS = [
    {"tag": "cls_P0", "file": "cls_P0.onnx", "config": "CAFE P0"},
    {"tag": "cls_P10", "file": "cls_P10.onnx", "config": "CAFE P10"},
    {"tag": "posterv2", "file": "posterv2.onnx", "config": "POSTERv2 (external)"},
]

RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
VDD_RE = re.compile(r"VDD_IN (\d+)mW")
ALT_VDD_RE = re.compile(r"VDD_GPU_SOC (\d+)mW")


class Tegrastats:
    def __init__(self, log: Path):
        self.log = log

    def __enter__(self):
        subprocess.run([TEGRASTATS, "--stop"], capture_output=True)
        self.fh = self.log.open("w")
        self.proc = subprocess.Popen(
            [TEGRASTATS, "--interval", str(SAMPLE_MS)],
            stdout=self.fh, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        return self

    def __exit__(self, *exc):
        subprocess.run([TEGRASTATS, "--stop"], capture_output=True)
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.fh.close()


def parse(log: Path) -> dict:
    ram, total, power = [], None, []
    for line in log.read_text(errors="ignore").splitlines():
        m = RAM_RE.search(line)
        if m:
            ram.append(int(m.group(1)))
            total = int(m.group(2))
        p = VDD_RE.search(line) or ALT_VDD_RE.search(line)
        if p:
            power.append(int(p.group(1)))
    if not ram:
        return {}
    out = {"ram_total_mb": total, "ram_peak_mb": max(ram),
           "ram_mean_mb": round(sum(ram) / len(ram), 1), "samples": len(ram)}
    if power:
        out["power_peak_mw"] = max(power)
        out["power_mean_mw"] = round(sum(power) / len(power), 1)
    return out


def run_model(model: dict) -> dict:
    path = ONNX_DIR / model["file"]
    log = ROOT / f"tegra_ort_{model['tag']}.log"

    with Tegrastats(log):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(path), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out_name = sess.get_outputs()[0].name
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        x = np.random.rand(*shape).astype(np.float32)

        for _ in range(WARMUP):
            sess.run([out_name], {inp.name: x})
        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            sess.run([out_name], {inp.name: x})
        latency_ms = (time.perf_counter() - t0) / ITERATIONS * 1000

        del sess

    res = parse(log)
    res["mean_latency_ms"] = round(latency_ms, 2)
    res["onnx_size_mb"] = round(path.stat().st_size / 1e6, 2)
    return res


def main() -> None:
    idle_log = ROOT / "tegra_ort_idle.log"
    print(f"idle baseline ({IDLE_SECONDS}s)...")
    with Tegrastats(idle_log):
        time.sleep(IDLE_SECONDS)
    idle = parse(idle_log)
    base = idle.get("ram_mean_mb", 0)
    print(f"  idle RAM {base:.0f}/{idle.get('ram_total_mb')} MB  "
          f"power {idle.get('power_mean_mw', 0):.0f} mW\n")

    report = {"runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider",
              "idle": idle, "models": {}}

    print(f"{'model':10s} {'onnx MB':>8s} {'peakRAM':>8s} {'delta':>7s} "
          f"{'x onnx':>7s} {'lat ms':>9s} {'peak mW':>8s}")
    for model in MODELS:
        if not (ONNX_DIR / model["file"]).is_file():
            print(f"{model['tag']:10s} SKIPPED")
            continue
        res = run_model(model)
        delta = res["ram_peak_mb"] - base
        res["peak_ram_delta_mb"] = round(delta, 1)
        res["peak_ram_vs_onnx_ratio"] = round(delta / res["onnx_size_mb"], 2)
        res["config"] = model["config"]
        report["models"][model["tag"]] = res
        print(f"{model['tag']:10s} {res['onnx_size_mb']:8.1f} "
              f"{res['ram_peak_mb']:8d} {delta:7.0f} "
              f"{res['peak_ram_vs_onnx_ratio']:7.1f} "
              f"{res['mean_latency_ms']:9.1f} {res.get('power_peak_mw', 0):8d}")

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
