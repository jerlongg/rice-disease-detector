# Rice Disease Classifier — MobileNetV3-Large Architecture

## Overview

| Item | Detail |
|---|---|
| Backbone | MobileNetV3-Large (ImageNet pretrained) |
| Classes | 6 — Bacterial\_Blight, Blast, Brown\_Spot, Tungro, Healthy\_Rice\_Leaf, Hispa |
| Input size | 224 × 224 × 3 |
| Total params | ~5.4M |
| Loss | CrossEntropyLoss (label\_smoothing=0.1) |
| Purpose | Lightweight comparison model vs ResNet-50 |

---

## What is MobileNetV3-Large?

MobileNetV3 is a neural network designed specifically for **mobile and edge devices**. Unlike ResNet-50 which prioritises accuracy, MobileNetV3 is engineered to be fast and small while staying competitive in accuracy.

It achieves this through two core ideas:

**Depthwise Separable Convolutions** — splits a standard convolution into two cheaper steps:
```
Standard conv  : filters entire image at once          → expensive
Depthwise conv : one filter per channel separately     → cheap
Pointwise conv : combines channel outputs with 1×1     → cheap
Total cost     : ~8–9× fewer operations than standard
```

**Squeeze-and-Excitation blocks** — a built-in lightweight attention mechanism inside each block that reweights channels based on global context. This is similar in spirit to CBAM but much lighter.

---

## Model Architecture

```
Input (B, 3, 224, 224)
    │
    ▼
features[0]   — Conv2d 3→16, stride 2       → (B, 16, 112, 112)
    │
    ▼
features[1]   — InvertedResidual block       → (B, 16, 112, 112)
    │
    ▼
features[2–3] — InvertedResidual blocks      → (B, 24, 56, 56)
    │
    ▼
features[4–6] — InvertedResidual + SE blocks → (B, 40, 28, 28)
    │
    ▼
features[7–12]— InvertedResidual + SE blocks → (B, 112, 14, 14)
    │
    ▼
features[13–15]—InvertedResidual + SE blocks → (B, 160, 7, 7)
    │
    ▼
features[16]  — Conv2d 160→960, BN, Hardswish → (B, 960, 7, 7)
    │
    ▼
AdaptiveAvgPool2d(1)                          → (B, 960, 1, 1)
    │
    ▼
Head
    Flatten          → (B, 960)
    Dropout(0.3)
    Linear(960, 128) + ReLU
    Dropout(0.2)
    Linear(128, 6)
    │
    ▼
Logits (B, 6)
```

---

## Key Architectural Differences vs ResNet-50

| | ResNet-50 | MobileNetV3-Large |
|---|---|---|
| Total params | ~25.6M | ~5.4M |
| Feature extractor | `children()[:-1]` | `backbone.features` |
| Built-in pooling | Yes — inside `children()[:-1]` | No — must add `AdaptiveAvgPool2d(1)` manually |
| Feature output size | 2048 | 960 |
| Head input | `Linear(2048, 128)` | `Linear(960, 128)` |
| Internal attention | None | Squeeze-and-Excitation in each block |
| Activation | ReLU | Hardswish (more efficient) |
| Speed | Slower | ~5× faster inference |
| Use case | High accuracy | Edge / mobile deployment |

---

## Classifier Head

Same structure as ResNet-50, only input size differs:

```
Flatten          →  (B, 960)
Dropout(0.3)     →  regularisation
Linear(960→128)  →  compress features
ReLU             →  non-linearity
Dropout(0.2)     →  regularisation
Linear(128→6)    →  one score per class
```

---

## Training Strategy — Two Phases

### Phase 1 — Head warm-up (10 epochs)

| Setting | Value |
|---|---|
| Frozen | Entire `features` (all 17 blocks) |
| Trainable | `pool` + `head` only |
| Optimizer | Adam |
| LR | 1e-3 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |

### Phase 2 — Fine-tuning (20 epochs max)

| Setting | Value |
|---|---|
| Unfrozen | `features[13]` through `features[16]` (last 4 blocks) |
| Frozen | `features[0]` through `features[12]` |
| Optimizer | Adam |
| LR | 1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | patience=5 on val\_loss |

**Why last 4 blocks?** MobileNetV3 has 17 feature blocks (0–16). The last 4 blocks (13–16) contain the highest-level semantic features — disease-level patterns, lesion textures. Earlier blocks contain low-level features (edges, colours) that are universal and should not be disturbed.

This mirrors the ResNet-50 strategy of unfreezing layer3 + layer4 — proportionally equivalent depth.

---

## Augmentation Pipeline

Identical to ResNet-50 — ensures fair comparison.

| Augmentation | Parameter | Purpose |
|---|---|---|
| Horizontal flip | p=0.5 | Orientation invariance |
| Rotation | ±15° | Field angle variation |
| Colour jitter + hue shift | brightness=0.2, hue±10 | Lighting variation |
| Random crop | padding=20 | Partial leaf / zoom variation |
| Shadow overlay | p=0.4 | Uneven field lighting |
| Solid-colour bg patch | p=0.3 | Blue fabric / wall background |
| JPEG compression | p=0.4, q=55–90 | Phone camera artifacts |

---

## Class Balancing

Same `WeightedRandomSampler` as ResNet-50:

```
sample_weight[i] = 1 / count(class[i])
```

Ensures equal sampling frequency across all 6 classes per epoch.

---

## Grad-CAM++ Hook

Different target layer from ResNet-50:

```python
# ResNet-50   → list(model.backbone.children())[7]   layer4
# MobileNetV3 → model.features[16]                   last conv block
target = model.features[16]   # output: (B, 960, 7, 7)
```

Both produce a 7×7 spatial heatmap — same resolution, different semantic depth.

---

## Full Architecture Summary

```
Input (B, 3, 224, 224)
    │
    ▼
MobileNetV3-Large features (17 blocks)
    ├─ blocks 0–12   FROZEN in Phase 2
    └─ blocks 13–16  UNFROZEN in Phase 2
    │   Output: (B, 960, 7, 7)
    ▼
AdaptiveAvgPool2d(1)     ← added manually
    │   Output: (B, 960, 1, 1)
    ▼
Head
    ├─ Flatten  →  (B, 960)
    ├─ Dropout(0.3)
    ├─ Linear 960 → 128 + ReLU
    ├─ Dropout(0.2)
    └─ Linear 128 → 6
    │   Output: (B, 6) logits
    ▼
Softmax → class probabilities
```

---

## Parameter Count

| Component | Params |
|---|---|
| features (all 17 blocks) | ~4.2M |
| AdaptiveAvgPool2d | 0 |
| Head | ~124K |
| **Total** | **~5.4M** |

Phase 1 trains ~124K params (head only).
Phase 2 trains ~1.5M params (last 4 blocks + head).

---

## Why MobileNetV3 for Comparison?

| Consideration | Detail |
|---|---|
| Speed | ~5× faster than ResNet-50 at inference |
| Size | ~5× smaller — deployable on a phone |
| Accuracy | Slightly lower than ResNet-50 in general benchmarks |
| Field use case | A farmer using a phone app needs fast, lightweight inference |

If MobileNetV3 achieves close accuracy to ResNet-50 with 5× fewer parameters, it is the better choice for real-world deployment on mobile devices — which is the ultimate goal of this project.
