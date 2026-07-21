# Datasets

Three datasets fill the three evaluation roles in CAFE. Only the stress set is versioned in this
repository; the two public benchmarks are not redistributed.

| Dataset | Role | In repo? |
|---|---|---|
| WIDER FACE | Stage 1 detector training + evaluation | No — download |
| FER2013 | Stage 2 classifier training + evaluation | No — download |
| In-the-wild stress set | End-to-end pipeline evaluation | **Yes** — `data/Stress_test/` |

Expected layout once both downloads are in place:

```
data/
├── FER2013/
│   ├── test/{angry,disgust,fear,happy,neutral,sad,surprise}/*.jpg
│   └── val -> test          (symlink created by the notebook; TensorRT INT8
│                             calibration defaults to the 'val' split)
├── wider_face_w_test/
│   ├── widerface_yolo.yaml  (nc: 1, names: [face])
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/     YOLO format, normalized xywh
└── Stress_test/
    ├── emotion_raw_images/  01.png … 20.png
    └── labels_out/          01.txt … 20.txt
```

## FER2013

Kaggle: *Challenges in Representation Learning: Facial Expression Recognition Challenge*
(Carrier & Courville, ICML 2013 workshop). 48×48 grayscale, seven emotion classes.

| Split | Images | Note |
|---|---|---|
| Train | 28,709 | class-imbalanced |
| Val | 3,589 | |
| Test | 3,589 | held-out evaluation |

Arrange as ImageFolder directories under `data/FER2013/`, one subdirectory per class, using the
class order the notebook expects:

```python
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
```

The class-index mapping is positional — reordering these directories silently corrupts the
reported top-1 accuracy.

The notebook creates `data/FER2013/val` as a symlink to `test/` because the Ultralytics INT8
calibrator defaults to the `val` split. This is deliberate: INT8 calibration for the classifier
uses the held-out FER2013 split, matching the paper's stated calibration protocol.

## WIDER FACE

Yang et al., CVPR 2016. Wide-field-of-view, dense, occluded, scale-varying face scenes —
the reason it is used here rather than a curated face-detection set.

| Split | Images | Face instances |
|---|---|---|
| Train | 12,880 | 131,551 |
| Val | 3,532 | 35,814 |
| Test | 3,226 | 32,510 |

Convert to YOLO format (single class, `nc: 1`, `names: [face]`) with normalized `xc yc w h`
label files mirroring the image tree, and write `widerface_yolo.yaml` alongside. Cell 6 asserts
`nc == 1` and will stop if the conversion produced a multi-class YAML.

Detector INT8 calibration uses the WIDER FACE **val** split; mAP50-95 is reported on **test**.

## In-the-wild stress set

Included: `data/Stress_test/`. 20 unconstrained multi-face scenes curated from the WIDER FACE test
split, with **132 hand-annotated face instances** — an average density of 6.6 faces per image.
Each instance carries a bounding box and a coarse affect label.

This set is a controlled deployment probe, not a population-scale FER benchmark. It is what
exposes multi-face inference latency, missed-detection penalties, crop-dependent classification
behavior, and cascading detector→classifier failures under edge constraints.

For the end-to-end evaluation the seven FER2013 emotion classes are mapped to three coarse
affective groups — **neutral, happy, sad**. This mapping applies only to the pipeline-level
evaluation, not to the CAFE methodology in general; full seven-class performance is reported at
the component level for the Stage 2 classifier. Detections are matched to ground truth greedily
at IoU ≥ 0.50; boxes below that threshold are false positives, and unmatched ground-truth boxes
are missed detections that penalize pipeline recall directly.
