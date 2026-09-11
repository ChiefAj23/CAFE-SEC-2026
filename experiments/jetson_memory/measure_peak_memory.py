"""Measured runtime memory and power on the Jetson Orin Nano.

The deployment argument prices classifier size at the largest single weight in
the ranking, and treats a 180.6 MB classifier as disqualifying against a 7.4 GB
shared envelope. That argument is made entirely from *serialized* model size,
which is not the same quantity as runtime memory pressure: a serialized file
does not include the execution context, the activation workspace, the CUDA
context, or the runtime's own allocations, and on a unified-memory device the
GPU allocations come out of the same pool as the operating system's.

Serialized size can therefore understate or overstate the real constraint, and
the paper does not establish which. This script measures it directly.

For each model it samples tegrastats while an inference workload runs under
TensorRT, and reports:

  - peak RAM against the device total, and the delta over an idle baseline
  - peak power draw, against the board's configured power mode
  - serialized ONNX size and built engine size, for comparison

Engines are built with trtexec, so the measurement needs no PyTorch on the
board -- only the TensorRT that ships with JetPack.

Usage (on the Jetson):
    python3 measure_peak_memory.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path.home() / "cafe_memtest"
ONNX_DIR = ROOT / "onnx"
ENGINE_DIR = ROOT / "engines"
OUT = ROOT / "jetson_memory.json"

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
TEGRASTATS = "/usr/bin/tegrastats"

SAMPLE_MS = 100
IDLE_SECONDS = 6
INFER_ITERATIONS = 300

# Workspace cap. The board has ~7.6 GB shared between OS and accelerator, so an
# unbounded builder workspace can itself exhaust the envelope we are measuring.
WORKSPACE_MB = 1024

# `shape` is set only for exports with a dynamic dimension. Passing --shapes
# for an input whose dimensions are already fixed makes trtexec fail config
# setup, so the static exports leave it None.
MODELS = [
    {"tag": "det_P0", "file": "det_P0.onnx", "shape": None,
     "role": "detector", "config": "P0"},
    {"tag": "det_P10", "file": "det_P10.onnx", "shape": None,
     "role": "detector", "config": "P10"},
    {"tag": "cls_P0", "file": "cls_P0.onnx", "shape": None,
     "role": "classifier", "config": "P0"},
    {"tag": "cls_P10", "file": "cls_P10.onnx", "shape": None,
     "role": "classifier", "config": "P10"},
    {"tag": "posterv2", "file": "posterv2.onnx", "shape": "images:1x3x224x224",
     "role": "classifier", "config": "POSTERv2 (external)"},
]

RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
VDD_RE = re.compile(r"VDD_IN (\d+)mW")
# Older/newer tegrastats builds label the rail differently.
ALT_VDD_RE = re.compile(r"VDD_GPU_SOC (\d+)mW")


class Tegrastats:
    """Sample tegrastats into a log for the duration of a context."""

    def __init__(self, log: Path):
        self.log = log
        self.proc = None

    def __enter__(self):
        self.fh = self.log.open("w")
        self.proc = subprocess.Popen(
            [TEGRASTATS, "--interval", str(SAMPLE_MS)],
            stdout=self.fh, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
        return self

    def __exit__(self, *exc):
        subprocess.run([TEGRASTATS, "--stop"], capture_output=True)
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        self.fh.close()


def parse(log: Path) -> dict:
    ram_used, ram_total, power = [], None, []
    for line in log.read_text(errors="ignore").splitlines():
        m = RAM_RE.search(line)
        if m:
            ram_used.append(int(m.group(1)))
            ram_total = int(m.group(2))
        p = VDD_RE.search(line) or ALT_VDD_RE.search(line)
        if p:
            power.append(int(p.group(1)))
    if not ram_used:
        return {}
    out = {
        "samples": len(ram_used),
        "ram_total_mb": ram_total,
        "ram_peak_mb": max(ram_used),
        "ram_mean_mb": round(sum(ram_used) / len(ram_used), 1),
        "ram_min_mb": min(ram_used),
    }
    if power:
        out["power_peak_mw"] = max(power)
        out["power_mean_mw"] = round(sum(power) / len(power), 1)
    return out


def build_engine(model: dict) -> tuple[Path | None, dict]:
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    engine = ENGINE_DIR / f"{model['tag']}.engine"
    onnx = ONNX_DIR / model["file"]
    info = {"engine_built": engine.is_file()}
    if engine.is_file():
        info["engine_size_mb"] = round(engine.stat().st_size / 1e6, 2)
        return engine, info

    log = ROOT / f"tegra_build_{model['tag']}.log"
    cmd = [
        TRTEXEC, f"--onnx={onnx}", f"--saveEngine={engine}", "--fp16",
        f"--memPoolSize=workspace:{WORKSPACE_MB}", "--skipInference",
    ]
    if model["shape"]:
        cmd.insert(4, f"--shapes={model['shape']}")
    t0 = time.perf_counter()
    with Tegrastats(log):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    info["build_seconds"] = round(time.perf_counter() - t0, 1)
    info["build_ok"] = proc.returncode == 0
    info["build_tegrastats"] = parse(log)
    if proc.returncode != 0:
        info["build_error_tail"] = proc.stderr.strip().splitlines()[-4:]
        return None, info
    info["engine_built"] = True
    info["engine_size_mb"] = round(engine.stat().st_size / 1e6, 2)
    return engine, info


def measure(model: dict, engine: Path) -> dict:
    log = ROOT / f"tegra_run_{model['tag']}.log"
    cmd = [
        TRTEXEC, f"--loadEngine={engine}",
        f"--iterations={INFER_ITERATIONS}", "--avgRuns=100", "--warmUp=1000",
    ]
    if model["shape"]:
        cmd.insert(2, f"--shapes={model['shape']}")
    with Tegrastats(log):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    res = parse(log)
    res["run_ok"] = proc.returncode == 0
    for line in proc.stdout.splitlines():
        if "Latency:" in line and "median" in line:
            res["trtexec_latency_line"] = line.strip()
        if "Throughput:" in line:
            res["trtexec_throughput_line"] = line.strip()
    return res


def main() -> None:
    if not Path(TRTEXEC).is_file():
        raise SystemExit(f"trtexec not found at {TRTEXEC}")

    report = {"device": {}, "idle": {}, "models": {}}

    # Board configuration, for the record.
    for key, cmd in (("power_mode", ["nvpmodel", "-q"]),
                     ("tegra_release", ["cat", "/etc/nv_tegra_release"])):
        try:
            report["device"][key] = subprocess.run(
                cmd, capture_output=True, text=True).stdout.strip().splitlines()[:2]
        except Exception:
            pass

    print(f"idle baseline ({IDLE_SECONDS}s)...")
    idle_log = ROOT / "tegra_idle.log"
    with Tegrastats(idle_log):
        time.sleep(IDLE_SECONDS)
    report["idle"] = parse(idle_log)
    idle_ram = report["idle"].get("ram_mean_mb", 0)
    total = report["idle"].get("ram_total_mb", 0)
    print(f"  idle RAM {idle_ram:.0f} / {total} MB   "
          f"power {report['idle'].get('power_mean_mw', 0):.0f} mW\n")

    for model in MODELS:
        onnx = ONNX_DIR / model["file"]
        if not onnx.is_file():
            print(f"{model['tag']:10s} SKIPPED (missing {model['file']})")
            continue
        entry = {"role": model["role"], "config": model["config"],
                 "onnx_size_mb": round(onnx.stat().st_size / 1e6, 2)}
        print(f"{model['tag']:10s} building engine "
              f"(onnx {entry['onnx_size_mb']:.1f} MB)...")
        engine, build_info = build_engine(model)
        entry.update(build_info)
        if engine is None:
            entry["verdict"] = "ENGINE BUILD FAILED"
            print(f"           BUILD FAILED: {build_info.get('build_error_tail')}")
            report["models"][model["tag"]] = entry
            continue

        run = measure(model, engine)
        entry["runtime"] = run
        peak = run.get("ram_peak_mb", 0)
        entry["peak_ram_delta_mb"] = round(peak - idle_ram, 1)
        entry["peak_ram_vs_onnx_ratio"] = (
            round(entry["peak_ram_delta_mb"] / entry["onnx_size_mb"], 2)
            if entry["onnx_size_mb"] else None
        )
        print(f"           engine {entry.get('engine_size_mb')} MB   "
              f"peak RAM {peak} / {total} MB   "
              f"delta over idle {entry['peak_ram_delta_mb']:.0f} MB   "
              f"peak power {run.get('power_peak_mw', 0):.0f} mW")
        report["models"][model["tag"]] = entry

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
