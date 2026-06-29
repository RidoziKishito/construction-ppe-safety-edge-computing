# Construction Site PPE Safety Detection — Edge Computing System

A real-time Personal Protective Equipment (PPE) detection and zone-based safety monitoring system built for edge deployment on construction sites. The system combines a lightweight YOLO-based detection model with an on-device inference pipeline to enforce PPE compliance without continuous cloud connectivity.

This project is developed as part of a university-level innovation competition focused on practical Edge Computing applications in AI and IoT.

***

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Detection Classes](#detection-classes)
- [Repository Structure](#repository-structure)
- [AI Model](#ai-model)
  - [Model Ablation Study](#model-ablation-study)
  - [Dataset](#dataset)
  - [Training on Modal](#training-on-modal)
  - [Export](#export)
- [Edge Pipeline](#edge-pipeline)
- [Zone-Based Safety Logic](#zone-based-safety-logic)
  - [Zone Configuration](#zone-configuration)
  - [Worker Positioning](#worker-positioning)
  - [Alert Classification](#alert-classification)
- [Performance Targets and Benchmarks](#performance-targets-and-benchmarks)
  - [Model Ablations](#model-ablations)
  - [ONNX Export and Quantization](#onnx-export-and-quantization)
  - [Temporal Smoothing stability](#temporal-smoothing-stability)
  - [Bandwidth Savings](#bandwidth-savings)
- [Setup and Usage](#setup-and-usage)
  - [Prerequisites](#prerequisites)
  - [Running the Edge Pipeline](#running-the-edge-pipeline)
  - [Running Unit Tests](#running-unit-tests)
  - [Running Training on Modal](#running-training-on-modal)
- [Team](#team)
- [License](#license)

***

## Overview

Construction sites are among the most hazardous working environments. PPE compliance — helmets, high-visibility vests, gloves, boots, and goggles — is the primary line of defense against injury. Manual inspection is inconsistent and resource-intensive.

This system addresses that gap through automated, context-aware detection that runs directly on edge hardware (targeting NVIDIA Jetson Xavier NX). Unlike cloud-only approaches that stream raw video continuously, the system performs inference on-device and transmits only structured alert events, reducing bandwidth consumption by over 99%.

The key research contribution over prior work is zone-aware risk classification: the system does not only ask "is the worker wearing PPE?" but also "what zone is the worker in, and what PPE is required there?" This enables tiered alerting — CRITICAL for workers in high-risk zones or missing mandatory PPE, and WARNING for general non-compliance.

***

## System Architecture

The system is divided into three layers that communicate along a single inference path:

```
Camera (RTSP / USB) 
        |
        v
  Edge Device (Jetson Xavier NX)
  +------------------------------------------+
  |  Video Capture --> YOLO Inference (ONNX) |
  |       |                                  |
  |  Detection Results (bounding boxes)      |
  |       |                                  |
  |  Zone Checker (point-in-polygon)         |
  |       |                                  |
  |  Rule Engine  --> Alert Events           |
  |  (NORMAL / WARNING / CRITICAL)           |
  +------------------------------------------+
        |
        v
  Edge Gateway (MQTT / WebSocket / Local Web Dashboard)
  +------------------------------------------+
  |  Live Web UI (Local network)             |
  |  Date-prefixed CSV logs                  |
  |  Violation image snapshots               |
  +------------------------------------------+
        |
        v
  Cloud Dashboard (alert log, compliance reports)
```

Only alert events and anonymized metadata are sent upstream. Raw video never leaves the edge device during normal operation.

***

## Detection Classes

The model is trained to detect 11 classes derived from the Construction-PPE dataset:

| ID | Class | Type |
|---|---|---|
| 0 | helmet | PPE present |
| 1 | gloves | PPE present |
| 2 | vest | PPE present |
| 3 | boots | PPE present |
| 4 | goggles | PPE present |
| 5 | none | No PPE visible |
| 6 | Person | Worker detected |
| 7 | no_helmet | PPE violation |
| 8 | no_goggle | PPE violation |
| 9 | no_gloves | PPE violation |
| 10 | no_boots | PPE violation |

Violation classes (no_*) are treated as high-priority signals in the rule engine.

***

## Repository Structure

```
construction-ppe-safety-edge-computing/
|
+-- ai-model/
|   +-- configs/
|   |   +-- data_modal.yaml          # Dataset config for Modal container mount
|   |   +-- yolo26n-cbam.yaml        # YOLO26n + CBAM attention architecture
|   |   +-- yolo26n-ghost.yaml       # YOLO26n + GhostConv lightweight variant
|   |
|   +-- docs/
|   |   +-- dataset-checklist.md     # Guidelines for data split & class balance
|   |   +-- edge_optimization_summary.md  # Deployment performance summary
|   |   +-- training-guide.md        # Guide to training configurations
|   |
|   +-- outputs/
|   |   +-- ablations/               # Output weight and metric files per ablation run
|   |   +-- edge_results/            # Latency, smoothing, and bandwidth CSVs
|   |   +-- RESULTS.md               # Summary of model and edge benchmarks
|   |   +-- final_ranked_table.csv   # Ranked ablation results table
|   |
|   +-- scripts/
|       +-- train_baseline.py        # Run 1: YOLO26n baseline training
|       +-- train_attention.py       # Run 2: YOLO26n + CBAM attention training
|       +-- train_ghost.py           # Run 3: YOLO26n + GhostConv training
|       +-- export_model.py          # Export trained weights to ONNX / TensorRT
|       +-- split_val.py             # Train/val/test split utility
|       +-- upload_dataset.py        # Upload local dataset to Modal Volume
|
+-- edge-pipeline/
|   +-- configs/
|   |   +-- ppe_classes.txt          # Detection class names list
|   |   +-- zones.json               # Zone definitions and coordinates
|   |
|   +-- tests/
|   |   +-- test_rule_engine.py      # Rule engine unit test suite
|   |
|   +-- edge_infer.py                # Pipeline entrypoint with temporal smoothing
|   +-- logger.py                    # CSV event logger and image snapshot saver
|   +-- rule_engine.py               # Point-in-polygon and PPE risk classification
|   +-- display.py                   # Bounding box & warning text drawing utility
|   +-- export_dummy_model.py        # Export placeholder ONNX for pipeline testing
|   +-- requirements.txt             # Edge pipeline Python dependencies
|
+-- .gitignore
+-- README.md
```

***

## AI Model

### Model Ablation Study

Seven model configurations were benchmarked to identify the best YOLO26n variant for edge deployment. All are based on YOLO26n (the nano variant of YOLO v26) trained from scratch on the Construction-PPE dataset. 

The complete ranked ablation results (all 2,663 validation images):

| Rank | Experiment | mAP@0.5 | Violation mAP@0.5 | Infer (ms/img) | Deploy recommendation |
|---|---|---|---|---|---|
| 1 | baseline_yolo26n | 0.677 | 0.580 | 2.93 | Edge deploy — fastest top-tier violation model |
| 2 | baseline_violation_oversample | 0.690 | 0.586 | 3.18 | Paper/report — best absolute mAP, slightly slower |
| 3 | baseline_violation_aug | 0.670 | 0.557 | 2.95 | Not recommended — augmentation underperforms |
| 4 | ghost_violation_oversample | 0.629 | 0.556 | 3.21 | Ablation only |
| 5 | cbam_violation_oversample | 0.642 | 0.553 | 3.15 | Ablation only |
| 6 | cbam_existing_repro | 0.618 | 0.525 | 3.15 | Not recommended |
| 7 | ghost_existing_repro | 0.613 | 0.518 | 3.11 | Not recommended |

CBAM (Run 5) inserts Convolutional Block Attention Modules after each C3k2 block in the YOLO neck and head. GhostConv (Run 4) replaces standard convolutions with Ghost modules. Baseline remains the deploy variant because attention and Ghost modules do not increase violation accuracy enough to justify the throughput overhead.

### Dataset

- Source: Construction-PPE dataset (Roboflow/Kaggle)
- Training images: 30,148
- Validation images: 143
- Test images: 141
- Classes: 11
- Image size during training: 640 x 640

The dataset includes both presence classes (helmet, vest, etc.) and explicit absence classes (no_helmet, no_goggle, etc.), enabling the rule engine to act on positive violation detections rather than relying solely on absence logic.

### Training on Modal

All training runs are executed on Modal cloud GPU infrastructure. The training scripts use Modal Functions with an NVIDIA L4 GPU (24 GB VRAM), a 6-hour timeout, and two persistent Modal Volumes:

- ppe-training-vol — stores checkpoints and training artifacts under /runs
- ppe-dataset-vol — stores the full dataset under /dataset

Step 1 — Upload the dataset (first time only):

```bash
cd ai-model/scripts
modal run upload_dataset.py
```

Step 2 — Run a training job:

```bash
# Baseline
modal run train_baseline.py

# CBAM attention variant
modal run train_attention.py

# GhostConv variant
modal run train_ghost.py
```

Step 3 — Download results:

```bash
modal volume get ppe-training-vol /runs ./local_runs
```

Training hyperparameters (common across runs):

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 64 |
| Optimizer | AdamW |
| Initial LR | 0.001 |
| LR final factor | 0.01 |
| Warmup epochs | 5 |
| Cosine LR decay | enabled |
| Label smoothing | 0.1 |
| Mosaic | 1.0 |
| Mixup | 0.15 |
| Copy-paste | 0.1 |
| Seed | 42 |

### Export

After training, export the best checkpoint to ONNX:

```bash
cd ai-model/scripts
python export_model.py --weights /path/to/best.pt --format onnx
```

***

## Edge Pipeline

The edge pipeline reads from a camera source (webcam index or RTSP URL / local video file), runs YOLO inference using the ONNX model, maps coordinates to safety zones, evaluates alert rules, filters warnings through temporal smoothing, and records violations.

The pipeline includes:
- edge_infer.py: Main orchestration loop and frame processing.
- rule_engine.py: Geofencing tests and alert level decision logic.
- logger.py: Records date-prefixed CSV alerts and exports JPEG frames showing critical violations.
- display.py: Live display annotation (bounding boxes and alert text).

***

## Zone-Based Safety Logic

Zone-based safety is the core differentiator of this system. Rather than applying a single global PPE rule, the system maps detections to named geofenced zones and applies zone-specific PPE requirements.

### Zone Configuration

Zones are defined as 2D polygons over the camera frame, stored in a JSON config file. The system features an **Interactive Zone Editor** integrated directly into the local dashboard, allowing users to draw perspective-accurate, multi-point polygons (e.g., trapezoids, custom shapes) over the live feed instead of simple rectangles:

```json
{
  "zones": [
    {
      "id": "Z01",
      "name": "Danger Zone (Lifting Area)",
      "type": "danger_zone",
      "polygon": [[100, 450], [650, 450], [650, 1000], [100, 1000]]
    },
    {
      "id": "Z02",
      "name": "Warning Zone (Scaffolding)",
      "type": "warning_zone",
      "polygon": [[700, 450], [1800, 450], [1800, 1000], [700, 1000]]
    }
  ]
}
```

### Worker Positioning

Zone membership is determined using two anchoring approaches:
1. **Person-centric positioning**: If a worker bounding box (`Person` class) is detected, the **foot point** (midpoint of the bottom edge: `(x1 + x2) // 2, y2`) is used.
2. **Standalone positioning**: If the detector misses the person box but catches a standalone PPE violation (e.g. `no_helmet`), the center of that violation box is used as the anchor point.

Point-in-polygon tests are performed using `cv2.pointPolygonTest`.

### Alert Classification

| Condition | Alert Level |
|---|---|
| Worker in a Danger Zone AND missing PPE | CRITICAL |
| Worker in a Warning Zone AND missing PPE | CRITICAL |
| Worker in a Danger Zone AND wearing all PPE | WARNING |
| Worker in a Warning Zone AND wearing all PPE | WARNING |
| Worker outside warning/danger zones AND missing PPE | WARNING |
| Worker outside warning/danger zones AND wearing all PPE | NORMAL |

***

## Performance Targets and Benchmarks

### Model Ablations
The `baseline_yolo26n` model provides a strong balance of latency and violation detection mAP (0.580), and serves as the deployment standard.

### ONNX Export and Quantization
Latency measurements taken on NVIDIA GPU environment (ONNX Runtime 1.26.0):

| Variant | Batch size | Model size | Latency (ms/img) | FPS | mean_violation_map50 |
|---|---|---|---|---|---|
| FP32 fixed B=1 | 1 | 9.4 MB | 8.99 | 111 | 0.660 |
| FP32 dynamic | 8 | 10.3 MB | 7.61 | 131 | 0.671 |
| FP16 dynamic | 4 | 5.7 MB | 9.05 | 110 | 0.671 |
| INT8 dynamic | 1 | 3.7 MB | 268.7 | 3.7 | 0.711 |

*Note: FP32 dynamic with batch size 8 is the deployment choice, delivering 131 FPS. FP16 dynamic is recommended for RAM-constrained platforms (5.7 MB size).*

### Temporal Smoothing stability
Temporal smoothing reduces frame-to-frame alert chatter by requiring the same violation to persist for N consecutive frames before logging:

| Video | Level | N consecutive frames | Total frames | Warning count | Critical count | Alert rate | FP reduction pct |
|---|---|---|---|---|---|---|---|
| demo_video3 | Frame-level | 1 | 153 | 17 | 0 | 11.11% | 0.0% |
| demo_video3 | Event-level | 3 | 153 | 2 | 0 | 1.31% | 88.2% |
| demo_video3 | Event-level | 5 | 153 | 1 | 0 | 0.65% | 94.1% |
| demo_video3 | Event-level | 10 | 153 | 1 | 0 | 0.65% | 94.1% |
| demo_video4 | Frame-level | 1 | 144 | 96 | 28 | 86.11% | 0.0% |
| demo_video4 | Event-level | 3 | 144 | 5 | 3 | 5.56% | 93.5% |
| demo_video4 | Event-level | 5 | 144 | 5 | 1 | 4.17% | 95.2% |
| demo_video4 | Event-level | 10 | 144 | 4 | 1 | 3.47% | 96.0% |

*Recommended setting: N=5 consecutive frames. This yields a 94.6% false-positive rate reduction while adding only 167 ms of latency (at 30 FPS).*

### Bandwidth Savings
Structured edge event logging versus raw video streaming:

| Scenario | Cloud strategy | Time period | Data size | Bandwidth reduction pct | Notes |
|---|---|---|---|---|---|
| Scenario A | Raw demo video streaming | 30 days | 1,399,160 MB | 0.0% | Weighted from demo videos |
| Scenario A reference | Continuous 1080p 2 Mbps stream | 30 days | 617,981 MB | 55.8% | Continuous raw streaming reference |
| Scenario B | Edge logs + violation snapshots | 30 days | 970 MB | 99.93% | 1.0 event/min, 500 B log, 15-30 KB snapshot |
| Scenario C | Logs + blurred critical frames | 30 days | 283 MB | 99.98% | 5.0 critical events/hour, 80 KB blurred JPEG |

***

## Setup and Usage

### Prerequisites

- Python 3.11
- OpenCV, Ultralytics, PyTorch, NumPy
- For training: a Modal account with GPU quota

### Running the Edge Pipeline

1. Install dependencies:
   ```bash
   pip install -r edge-pipeline/requirements.txt
   ```

2. Run pipeline using an ONNX model and video file:
   ```bash
   python edge-pipeline/edge_infer.py --model ai-model/best.onnx --source edge-pipeline/media/videos/input/demo_video3.mp4 --zones edge-pipeline/configs/zones/demo_video3.json --smooth 5 --inference-interval 3 --alert-cooldown 5
   ```

Command-line Arguments:
- --model: path to the YOLO detection model (ONNX format).
- --source: path to local mp4 file, RTSP URL, or webcam index (e.g., 0).
- --conf: confidence threshold (default: 0.25).
- --iou: NMS IOU threshold (default: 0.45).
- --zones: path to JSON zone coordinates file.
- --smooth: minimum consecutive frames to trigger an alert (default: 5).

Press `q` or close the preview window to stop execution. Logs are saved to `edge-pipeline/logs/` as dated CSV files, with violation screenshots saved in `edge-pipeline/logs/snapshots/`.

### Running Unit Tests

To run the rule engine and polygon test assertions:

```bash
python -m unittest edge-pipeline/tests/test_rule_engine.py
```

### Running Training on Modal

1. Authenticate with Modal:
   ```bash
   pip install modal
   modal setup
   ```

2. Upload raw images/labels to the volume:
   ```bash
   modal run ai-model/scripts/upload_dataset.py
   ```

3. Start training:
   ```bash
   modal run ai-model/scripts/train_baseline.py
   ```

***

## Team

**Trinity** — Edge Computing (FAE), 2026

| Name | Role | Responsibility |
|---|---|---|
| Nguyen Nhat Phat | AI / ML Lead | Model architecture, CBAM/Ghost training, quantization, benchmark |
| Tran Quoc Huy | Edge Systems Lead | Jetson deployment, pipeline, zone engine, MQTT gateway |
| Le Huu Truc | Dashboard and Documentation | Dataset annotation, web UI, reports, slides, demo video |

***

## License

This repository is maintained for academic and competition purposes. All rights reserved by the team.
