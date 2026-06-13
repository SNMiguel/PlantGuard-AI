# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Setup

```bash
pip install -r requirements.txt
```

The dataset (PlantVillage) is expected at `data/raw/plantvillage dataset/color/` — one subdirectory per class (e.g., `Apple___Apple_scab/`). The zip archive `plantvillage-dataset.zip` is in the repo root if it needs to be extracted.

## Commands

```bash
# Train with defaults (resnet18, 10 epochs, batch 32)
python main.py --data_dir "data/raw" --epochs 10 --batch_size 32 --model resnet18

# Freeze backbone (train classifier head only — faster, less data needed)
python main.py --data_dir "data/raw" --freeze_backbone

# Other model variants
python main.py --model resnet34
python main.py --model resnet50

# Run self-tests in individual modules
python models/resnet_model.py
python utils/data_loader.py
python utils/transforms.py
python utils/metrics.py
python utils/visualizer.py

# Inference + agent self-tests
python predict.py path/to/leaf.jpg        # single-image diagnosis (CLI)
python foundry.py                          # Foundry IQ grounding self-test (offline)
python foundry.py --selftest-live          # ping the configured KB, dump raw response
python agent.py                            # multi-step reasoning self-test

# Run the Gradio demo app (upload a leaf -> diagnosis + agent reasoning)
python app.py
```

Key CLI flags: `--save_dir` (default: `models/saved`), `--verbose`.

On Windows the entry-point scripts (`predict.py`, `app.py`) force UTF-8 stdout because the existing modules print Unicode glyphs (e.g. `✓`) that the default `cp1252` console can't encode.

## Architecture

This is a PyTorch transfer-learning pipeline for classifying 38 plant diseases across 14 species.

**Pipeline (driven by `main.py` in 6 steps):**

```
data/raw/  →  PlantDiseaseDataset  →  PlantDiseaseTransforms  →  ResNetClassifier
               (data_loader.py)        (transforms.py)             (resnet_model.py)
                70/15/15 split          224×224, ImageNet norm
                                                ↓
                                           Trainer  (train.py)
                                         Adam + ReduceLROnPlateau
                                         saves best → models/saved/
                                                ↓
                                      MetricsCalculator  +  Visualizer
                                        (metrics.py)       (visualizer.py)
                                      results/visualizations/
```

**Key classes:**
- `ResNetClassifier` (`models/resnet_model.py`) — wraps pretrained ResNet, replaces the FC head for N classes. Provides `save_checkpoint`/`load_checkpoint`, `predict` (returns softmax probs), `get_feature_maps`, `freeze_backbone`/`unfreeze_backbone`.
- `Trainer` (`train.py`) — owns the training loop, optimizer, scheduler, and best-checkpoint logic.
- `PlantDiseaseDataset` (`utils/data_loader.py`) — reads class folders, performs the 70/15/15 split, returns DataLoaders. `SingleImageDataset` is used for single-image inference.
- `PlantDiseaseTransforms` (`utils/transforms.py`) — provides `get_train_transforms()` (augmentation + normalize) and `get_val_transforms()` (resize + normalize only); `get_test_transforms()` aliases the latter.
- `MetricsCalculator` (`utils/metrics.py`) — accumulates per-batch predictions and computes accuracy, precision, recall, F1 (macro + weighted), and the confusion matrix.
- `Visualizer` (`utils/visualizer.py`) — plots training history, normalized confusion matrix, and sample predictions with probabilities.

## Inference + reasoning agent (Microsoft Agents League — Reasoning Agents track)

A second pipeline turns a single leaf image into a grounded, multi-step action plan:

```
leaf image → PlantDoctor (predict.py) → ReasoningAgent (agent.py) → AgentReport
              ResNet18 + checkpoint        4-step reasoning            (UI / dict)
              38-class label + conf         grounded by ↓
                                          FoundryIQ (foundry.py)
                                        cited agronomy knowledge
```

- `PlantDoctor` (`predict.py`) — loads `models/saved/resnet18_best.pth` via `ResNetClassifier` and classifies a path/PIL/array into one of 38 classes. `CLASS_NAMES` is hardcoded (matches `ImageFolder`'s alphabetical sort) so inference needs no dataset on disk. `get_doctor()` caches a singleton; CLI: `python predict.py leaf.jpg`.
- `ReasoningAgent` (`agent.py`) — takes the classifier label + confidence and reasons through **diagnosis → severity → treatment → economic impact**. Calls an Azure AI Foundry / OpenAI-compatible chat endpoint when configured, else composes the report from the grounded knowledge base (always demoable). Confidence modulates severity.
- `FoundryIQ` (`foundry.py`) — the Microsoft IQ grounding layer. Queries a live Foundry IQ retrieval endpoint when configured, else grounds answers against a curated, citation-backed agronomy KB (UC IPM, Cornell, PennState, CABI, FAO, APS). Returns passages + citations.
- `app.py` — Gradio UI (`gradio==4.7.1`): upload a leaf → diagnosis card (crop, condition, confidence, severity badge) + the agent's 4 reasoning steps + grounded sources.
- `config.py` — resolves LLM / Foundry IQ settings from env vars (or an optional `.env`; see `.env.example`). Nothing is required — absent credentials trigger the offline fallbacks.

**Marketing site:** `website/` — single-page static site (`index.html` + `styles.css` + `script.js`, no build step). Forest-green palette; references `website/assets/images/{hero-banner,hero-background,ai-scan}.png`. Open `website/index.html` directly in a browser.

## Platform note

`num_workers=0` is set throughout DataLoader calls for Windows compatibility. Do not change this to a non-zero value on Windows.
