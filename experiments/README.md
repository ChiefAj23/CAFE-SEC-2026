# Experiments

Validation experiments backing the paper's claims. Each subdirectory is self-contained and carries
its own README explaining why the experiment exists, what it does, and what it establishes. The
table in the top-level README maps each paper table and section to the directory that produces it.

| Experiment | What it measures | Needs |
|---|---|---|
| [`edus_recompute/`](edus_recompute/) | Whether the ranking depends on the EDUS weights, and the accuracy-ceiling bound for large classifiers | CPU only |
| [`precision_ablation/`](precision_ablation/) | How much of the pipeline speedup comes from compilation and FP16 versus from pruning | CUDA GPU + TensorRT |
| [`external_baselines_audit/`](external_baselines_audit/) | The input normalization and output class ordering each external classifier was trained with | CPU only, FER2013 |
| [`e2e_harness/`](e2e_harness/) | The end-to-end pipeline sweep, and calibration of the padding coefficient lambda | CUDA GPU + TensorRT |
| [`camera_ready/`](camera_ready/) | Per-scene sweep, the stress-set bootstrap, the EDUS sensitivity table, and the end-to-end ranking | CUDA GPU + TensorRT; recompute is CPU only |
| [`jetson_memory/`](jetson_memory/) | Real runtime memory and power against serialized model size | Jetson Orin Nano |
| [`surgery_equivalence/`](surgery_equivalence/) | Which routing-metadata fields the forward pass actually consumes | CPU only |

The CPU-only experiments run anywhere. The rest need the hardware named above.

---

## Environment

Python 3.10 or newer. The versions below are what the measurements were taken with; the TensorRT
pin is not optional.

```bash
python3 -m venv ~/cafe-env
~/cafe-env/bin/pip install -U pip wheel

# CUDA 12.8 wheels. Blackwell-generation cards are compute capability 12.0 and
# have no kernels in the cu126 builds.
~/cafe-env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

~/cafe-env/bin/pip install ultralytics==8.4.14 onnx onnxruntime torchinfo \
    opencv-python-headless pandas pyyaml scipy python-docx

# Must be 10.x. See the note below.
~/cafe-env/bin/pip install "tensorrt==10.15.1.29"
```

Verify before running anything:

```bash
~/cafe-env/bin/python -c "
import torch, ultralytics, tensorrt
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
print('ultralytics', ultralytics.__version__)
print('tensorrt', tensorrt.__version__)
print('EXPLICIT_BATCH', hasattr(tensorrt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'))
"
```

`EXPLICIT_BATCH` must print `True`.

### Things that will otherwise cost you an afternoon

**TensorRT must be 10.x, not 11.x.** TensorRT 11 removed
`NetworkDefinitionCreationFlag.EXPLICIT_BATCH`, which Ultralytics 8.4.14 still calls. Every engine
export fails with `AttributeError` on that attribute. Pinning to 10.15.1.29 also matches the
version the original desktop measurements used.

**Install plain `onnxruntime`, not `onnxruntime-gpu`,** unless the CUDA runtime on the machine
matches what the wheel was built against. The GPU wheel fails at import with
`ImportError: libcudart.so.13: cannot open shared object file` on a CUDA 12 system. The ONNX paths
in these experiments are only used for the external baselines, where CPU execution is sufficient
and is in fact what makes the classifier comparison fair.

**Engines are not portable.** A TensorRT `.engine` is tied to the GPU, driver, and TensorRT version
that built it. All scripts compile their own into `/tmp` and reuse them within a run; delete
`/tmp/cafe_*_engines` when changing hardware or TensorRT versions.

---

## Data

Only the stress set is versioned here. Both public datasets must be fetched separately, and
`docs/DATA.md` describes the expected layout.

**FER2013** is required by the audit, the ablation, and the end-to-end reproduction. Extract it so
the test split sits at `data/FER2013/test/<class>/*.jpg`, one directory per class, using exactly
these names:

```
angry  disgust  fear  happy  neutral  sad  surprise
```

The class-index mapping is positional. Renaming or reordering these directories silently changes
every reported accuracy.

The held-out split is identified by filename prefix: files named `PrivateTest_*` are the 3589-image
split the paper evaluates on, and the scripts filter on that prefix. Both `data/FER2013/val` and
`data/FER2013/train` are created as symlinks to `test/` on first use, because the Ultralytics
classification loader requires both to exist before it will run INT8 calibration.

**WIDER FACE** is not required by any experiment here. Detector mAP is carried over from the
published summary rather than recomputed, and the INT8 detector engines are calibrated on the
stress-set scenes instead of the WIDER FACE validation split. Both substitutions are recorded in
the output metadata.

**Large ONNX baselines.** Three external model exports exceed GitHub's file-size limit and are
excluded by `.gitignore`: RT-DETR (266 MB), POSTERv2 (181 MB), and PCNN (104 MB). See
`docs/EXTERNAL_BASELINES.md`. Scripts that need a missing file print `SKIP` or `NOT FOUND` and
continue, so everything else still runs without them. Drop them into `artifacts/` to get the full
result set.

---

## Running

From the repository root, with the virtualenv's Python.

```bash
# CPU only, seconds to minutes
~/cafe-env/bin/python experiments/edus_recompute/recompute_edus.py
~/cafe-env/bin/python experiments/edus_recompute/accuracy_headroom.py
~/cafe-env/bin/python experiments/surgery_equivalence/verify_metadata_consumption.py \
    --weights weights/P0/classifier.pt --task classify --imgsz 224

# CPU only, needs FER2013. Tens of minutes; POSTERv2 dominates the runtime.
~/cafe-env/bin/python experiments/external_baselines_audit/audit_preprocessing.py
~/cafe-env/bin/python experiments/external_baselines_audit/audit_all_baselines.py
~/cafe-env/bin/python experiments/external_baselines_audit/audit_norm_and_order.py

# GPU + TensorRT. First run compiles engines, which takes a few minutes each.
~/cafe-env/bin/python experiments/precision_ablation/ablate_precision.py
~/cafe-env/bin/python experiments/precision_ablation/ablate_precision.py --limit 200   # quick check

# GPU + TensorRT, needs FER2013. Roughly an hour for all 88 configurations.
cd experiments/e2e_harness
~/cafe-env/bin/python calibrate_lambda.py          # recovers lambda and the class mapping
~/cafe-env/bin/python run_e2e_sweep.py --lam 0.25 --surprise-to neutral \
    --out ../../results/validation/e2e_sweep.csv
```

The camera-ready tables are rebuilt from the recorded sweeps. The first needs the GPU; the last
two are CPU-only recomputations:

```bash
~/cafe-env/bin/python experiments/camera_ready/run_e2e_perscene.py
~/cafe-env/bin/python experiments/camera_ready/build_table8_camera_ready.py
~/cafe-env/bin/python experiments/camera_ready/sensitivity_camera_ready.py
~/cafe-env/bin/python experiments/camera_ready/bootstrap_stress_set.py
```

Long runs are best detached, since engine compilation and the 88-configuration sweep both outlast a
typical SSH session:

```bash
(setsid nohup ~/cafe-env/bin/python run_e2e_sweep.py --lam 0.25 --surprise-to neutral \
    --out out.csv > sweep.log 2>&1 &)
```

Avoid `pkill -f run_e2e_sweep` to stop it. The pattern matches the launching shell's own command
line, so the shell kills itself and the job never starts. Use `pgrep -f '[r]un_e2e_sweep'` to find
the real process id first.

The Jetson scripts run on the board itself and need no PyTorch there, only the TensorRT that ships
with JetPack. See [`jetson_memory/README.md`](jetson_memory/).

---

## Output

Per-experiment records write to `results/validation/`, and the camera-ready table scripts write to
`results/camera_ready/`. Nothing overwrites the platform summaries at the top of `results/`. Each end-to-end run also writes a `.meta.json` recording the parameters used and any
deviation from the original protocol, so results remain interpretable after the fact.

Hardware-dependent numbers will not match this repository's recorded values on different hardware.
Latency is hardware-specific by nature; accuracy figures and the ratios each experiment reports are
the portable quantities, and each experiment's README states which of its results are which.
