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
  - [Model Variants](#model-variants)
  - [Dataset](#dataset)
  - [Training on Modal](#training-on-modal)
  - [Export](#export)
- [Edge Pipeline](#edge-pipeline)
- [Zone-Based Safety Logic](#zone-based-safety-logic)
- [Performance Targets](#performance-targets)
- [Setup and Usage](#setup-and-usage)
  - [Prerequisites](#prerequisites)
  - [Running the Edge Pipeline](#running-the-edge-pipeline)
  - [Running Training on Modal](#running-training-on-modal)
- [Team](#team)
- [License](#license)

***

## Overview

Construction sites are among the most hazardous working environments. PPE compliance — helmets, high-visibility vests, gloves, boots, and goggles — is the primary line of defense against injury. Manual inspection is inconsistent and resource-intensive.

This system addresses that gap through automated, context-aware detection that runs directly on edge hardware (targeting NVIDIA Jetson Xavier NX). Unlike cloud-only approaches that stream raw video continuously, the system performs inference on-device and transmits only structured alert events, reducing bandwidth consumption by over 90%.

The key research contribution over prior work is **zone-aware risk classification**: the system does not only ask "is the worker wearing PPE?" but also "what zone is the worker in, and what PPE is required there?" This enables tiered alerting — CRITICAL for workers in high-risk zones with missing mandatory PPE, WARNING for general non-compliance, and ZONE VIOLATION for unauthorized zone entry regardless of PPE state.

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
  |  (NORMAL / WARNING / CRITICAL /          |
  |   ZONE_VIOLATION)                        |
  +------------------------------------------+
        |
        v
  Edge Gateway (MQTT / WebSocket)
        |
        v
  Cloud Dashboard (alert log, compliance reports)
```

Only alert events and anonymized metadata are sent upstream. Raw video never leaves the edge device during normal operation.

***

## Detection Classes

The model is trained to detect 11 classes derived from the Construction-PPE dataset:

| ID | Class      | Type              |
|----|------------|-------------------|
| 0  | helmet     | PPE present       |
| 1  | gloves     | PPE present       |
| 2  | vest       | PPE present       |
| 3  | boots      | PPE present       |
| 4  | goggles    | PPE present       |
| 5  | none       | No PPE visible    |
| 6  | Person     | Worker detected   |
| 7  | no_helmet  | PPE violation     |
| 8  | no_goggle  | PPE violation     |
| 9  | no_gloves  | PPE violation     |
| 10 | no_boots   | PPE violation     |

Violation classes (`no_*`) are treated as high-priority signals in the rule engine.

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
|   +-- scripts/
|       +-- train_baseline.py        # Run 1: YOLO26n baseline training
|       +-- train_attention.py       # Run 2: YOLO26n + CBAM attention training
|       +-- train_ghost.py           # Run 3: YOLO26n + GhostConv training
|       +-- export_model.py          # Export trained weights to ONNX / TensorRT
|       +-- split_val.py             # Train/val/test split utility
|       +-- upload_dataset.py        # Upload local dataset to Modal Volume
|
+-- edge-pipeline/
|   +-- video_pipeline.py            # Real-time inference loop (OpenCV + YOLO ONNX)
|   +-- export_dummy_model.py        # Export placeholder ONNX for pipeline testing
|   +-- requirements.txt             # Edge pipeline Python dependencies
|
+-- .gitignore
+-- README.md
```

***

## AI Model

### Model Variants

Three model configurations are benchmarked in this project. All are based on YOLO26n (the nano variant of YOLO v26) trained from scratch on the Construction-PPE dataset.

| Run | Script                | Architecture                  | Purpose                                     |
|-----|-----------------------|-------------------------------|---------------------------------------------|
| 1   | `train_baseline.py`   | YOLO26n (standard)            | Baseline reference for ablation comparison  |
| 2   | `train_attention.py`  | YOLO26n + CBAM                | Improved accuracy on small PPE objects      |
| 3   | `train_ghost.py`      | YOLO26n + GhostConv           | Reduced model size and inference latency    |

**CBAM (Run 2)** inserts Convolutional Block Attention Modules after each C3k2 block in the YOLO neck and head. Channel attention and spatial attention are applied sequentially. This is particularly effective for detecting small objects such as gloves and goggles in cluttered backgrounds (Woo et al., ECCV 2018).

**GhostConv (Run 3)** replaces standard convolutions with Ghost modules that generate feature maps using cheaper linear operations, reducing floating-point operations while retaining comparable accuracy. This variant targets deployment on constrained Jetson hardware.

### Dataset

- Source: Construction-PPE dataset (Roboflow/Kaggle)
- Training images: 30,148
- Validation images: 143
- Test images: 141
- Classes: 11 (see Detection Classes above)
- Image size during training: 640 x 640

The dataset includes both presence classes (`helmet`, `vest`, etc.) and explicit absence classes (`no_helmet`, `no_goggle`, etc.), enabling the rule engine to act on positive violation detections rather than relying solely on absence logic.

### Training on Modal

All training runs are executed on [Modal](https://modal.com) cloud GPU infrastructure. The training scripts use Modal Functions with an NVIDIA L4 GPU (24 GB VRAM), a 6-hour timeout, and two persistent Modal Volumes:

- `ppe-training-vol` — stores checkpoints and training artifacts under `/runs`
- `ppe-dataset-vol` — stores the full dataset under `/dataset`

**Step 1 — Upload the dataset (first time only):**

```bash
cd ai-model/scripts
modal run upload_dataset.py
```

**Step 2 — Run a training job:**

```bash
# Baseline
modal run train_baseline.py

# CBAM attention variant
modal run train_attention.py

# GhostConv variant
modal run train_ghost.py
```

**Step 3 — Download results:**

```bash
modal volume get ppe-training-vol /runs ./local_runs
```

Training hyperparameters (common across runs):

| Parameter        | Value     |
|------------------|-----------|
| Epochs           | 50        |
| Batch size       | 64        |
| Optimizer        | AdamW     |
| Initial LR       | 0.001     |
| LR final factor  | 0.01      |
| Warmup epochs    | 5         |
| Cosine LR decay  | enabled   |
| Label smoothing  | 0.1       |
| Mosaic           | 1.0       |
| Mixup            | 0.15      |
| Copy-paste       | 0.1       |
| Seed             | 42        |

### Export

After training, export the best checkpoint to ONNX (for edge deployment) or TensorRT (for Jetson INT8 acceleration):

```bash
cd ai-model/scripts
python export_model.py --weights /path/to/best.pt --format onnx
```

The exported ONNX model is consumed directly by `edge-pipeline/video_pipeline.py`.

***

## Edge Pipeline

The edge pipeline (`edge-pipeline/video_pipeline.py`) reads from a camera source (webcam index or RTSP URL), runs YOLO inference using the ONNX model via the Ultralytics runtime, and renders bounding boxes on the live frame.

Current state: person detection is active on the live feed. Zone checking and rule engine integration are in progress (next sprint).

**Run locally for testing:**

```bash
cd edge-pipeline
pip install -r requirements.txt
python video_pipeline.py
```

Press `q` or close the window to exit.

The pipeline window is named "Trinity Edge - Safety Pipeline". By default it opens the system webcam (`source=0`). To use a video file or RTSP stream, pass the path or URL as `video_source` in `run_pipeline()`.

***

## Zone-Based Safety Logic

Zone-based safety is the core differentiator of this system over standard PPE detectors. Rather than applying a single global PPE rule to all workers, the system maps each detected person to a named zone and applies zone-specific PPE requirements.

### Zone Configuration

Zones are defined as 2D polygons over the camera frame, stored in a JSON or YAML config file:

```yaml
zones:
  - name: high_risk_zone
    level: CRITICAL
    polygon: [[120, 80], [420, 80], [420, 350], [120, 350]]
    required_ppe: [helmet, vest, harness]

  - name: machine_zone
    level: CRITICAL
    polygon: [[430, 90], [700, 90], [700, 400], [430, 400]]
    required_ppe: [helmet, vest, boots, goggles]

  - name: restricted_zone
    level: ZONE_VIOLATION
    polygon: [[50, 400], [250, 400], [250, 550], [50, 550]]
    required_ppe: []

  - name: general_zone
    level: WARNING
    polygon: []
    required_ppe: [helmet, vest]
```

### Person-to-Zone Mapping

Zone membership is determined by checking whether the **foot point** of a worker's bounding box — the midpoint of the bottom edge — falls inside a zone polygon. The foot point is used rather than the bounding box center because zone hazards relate to where the worker is physically standing.

```python
foot_x = (x1 + x2) // 2
foot_y = y2
zone = point_in_zone(foot_x, foot_y, zone_configs)
```

Point-in-polygon testing uses the standard ray-casting algorithm.

### Alert Classification

| Condition                                              | Alert Level    |
|--------------------------------------------------------|----------------|
| Worker in danger zone, missing required PPE            | CRITICAL       |
| Worker in general zone, missing PPE                    | WARNING        |
| Worker in restricted zone (regardless of PPE)          | ZONE_VIOLATION |
| Worker with full PPE in permitted zone                 | NORMAL         |

***

## Performance Targets

| Metric                        | Target              |
|-------------------------------|---------------------|
| mAP@0.5 (test set)            | >= 90%              |
| End-to-end latency            | < 100 ms/frame      |
| FPS (Jetson Xavier NX)        | 10 - 15 FPS         |
| Bandwidth saving vs. cloud    | >= 90%              |
| False positive rate           | < 5%                |
| Exported model size           | < 50 MB             |

***

## Setup and Usage

### Prerequisites

- Python 3.11
- For training: a Modal account with GPU quota (`pip install modal`)
- For edge pipeline: OpenCV, Ultralytics (`pip install -r edge-pipeline/requirements.txt`)
- For Jetson deployment: JetPack 5.x, TensorRT 8.x (future sprint)

### Running the Edge Pipeline

```bash
# Install dependencies
pip install -r edge-pipeline/requirements.txt

# Run with webcam
python edge-pipeline/video_pipeline.py

# Run with video file (edit video_source inside the script)
# video_source = "path/to/video.mp4"
```

### Running Training on Modal

```bash
# Install Modal
pip install modal
modal setup

# Upload dataset to Modal Volume (once)
modal run ai-model/scripts/upload_dataset.py

# Train baseline
modal run ai-model/scripts/train_baseline.py

# Train CBAM variant
modal run ai-model/scripts/train_attention.py

# Train GhostConv variant
modal run ai-model/scripts/train_ghost.py

# Export model
python ai-model/scripts/export_model.py
```

***

## Team

**Trinity** — Edge Computing (FAE), 2026

| Name               | Role                          | Responsibility                                                 |
|--------------------|-------------------------------|----------------------------------------------------------------|
| Nguyen Nhat Phat   | AI / ML Lead                  | Model architecture, CBAM/Ghost training, quantization, benchmark |
| Tran Quoc Huy      | Edge Systems Lead             | Jetson deployment, pipeline, zone engine, MQTT gateway         |
| Le Huu Truc        | Dashboard and Documentation   | Dataset annotation, web UI, reports, slides, demo video        |

***

## License

This repository is maintained for academic and competition purposes. All rights reserved by the team.