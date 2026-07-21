# Environment

All reported measurements come from two hardware platforms. Batch size is 1 throughout, modelling
single-frame AII inference; latency values are averaged over the full test sets.

## Desktop profiling platform

| Component | Version |
|---|---|
| GPU | NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU, 25.1 GB VRAM |
| Python | 3.10.19 |
| PyTorch | 2.10.0 (CUDA 12.8) |
| TensorRT | 10.15.1.29 |
| Ultralytics | 8.4.14 |
| ONNX Runtime | 1.23.2 |

## Edge deployment platform

| Component | Version |
|---|---|
| Board | NVIDIA Jetson Orin Nano Developer Kit, 7.4 GB shared RAM |
| GPU | 1024-core NVIDIA Ampere @ 624 MHz |
| CPU | 6-core ARM Cortex-A78AE @ 1510 MHz |
| Power mode | 15 W |
| Python | 3.10.20 |
| PyTorch | 2.4.0 (NVIDIA JetPack NV24.05) |
| TensorRT | 8.6.2 |
| Ultralytics | 8.4.37 |
| ONNX Runtime | 1.18.0 |

## On the version gap

The two stacks differ because Jetson PyTorch builds lag mainline releases and different TensorRT
iterations apply distinct optimization passes. This is a property of the edge-hardware release
cycle, not an inconsistency in the experiment.

The consequence that matters: **TensorRT engines are compiled independently on each device and are
not portable.** An `.engine` built on the desktop will not load on the Jetson, and engines are
invalidated by TensorRT or driver upgrades on the same machine. Every `.engine` in this project is
a build artifact — none are versioned here, and `.gitignore` excludes them. Recompile on the
target device before benchmarking.

The portable reference paths are the native PyTorch FP32 `.pt` models. Any FP16 or INT8 number is
tied to the stack above.

## ONNX exports

External baselines are exported at **opset 20**, the highest version supported across the
deployment runtime stack on the Jetson Orin Nano. This opset supports the attention-related
operators used by the model inventory without operator decomposition — decomposition would modify
the computation graph and break measurement consistency across architectures.

## Reported footprints

Model footprints follow IEEE 754 width conventions: 4, 2, and 1 byte per parameter for FP32, FP16,
and INT8 respectively. External ONNX references are reported at their serialized size.
