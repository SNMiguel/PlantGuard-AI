# 🌿 PlantGuard-AI

**A reasoning agent that turns a single crop-leaf photo into a grounded, cited action plan — diagnosis, differential reasoning, treatment, and economic impact.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![Accuracy](https://img.shields.io/badge/Vision_Accuracy-95.02%25-brightgreen)
![Foundry IQ](https://img.shields.io/badge/Microsoft-Foundry_IQ-0078D4)
![License](https://img.shields.io/badge/License-MIT-green)

> **Microsoft Agents League — Reasoning Agents track.** Integrates **Microsoft Foundry IQ** (the required Microsoft IQ intelligence layer) for agentic, cited knowledge retrieval.

<p align="center">
  <a href="https://www.youtube.com/watch?v=alXvoNP0Eg4">
    <img src="https://img.youtube.com/vi/alXvoNP0Eg4/maxresdefault.jpg" alt="Watch the PlantGuard-AI demo" width="640" />
  </a>
  <br/>
  <a href="https://www.youtube.com/watch?v=alXvoNP0Eg4"><b>▶ Watch the 3-minute demo</b></a>
</p>

---

## 🎯 What it is

A diagnosis alone doesn't tell a farmer what to *do*. PlantGuard-AI pairs a **95%-accurate vision model** with a **multi-step reasoning agent** that:

1. **Detects** the disease from a leaf photo (ResNet18, 38 classes / 14 crops).
2. **Reasons through a differential diagnosis** when the call is ambiguous — running *separate Foundry IQ retrievals per candidate*, weighing distinguishing symptoms, and telling the farmer what to physically inspect to confirm.
3. **Grounds every recommendation** in cited agricultural knowledge via **Microsoft Foundry IQ** (Azure AI Search agentic retrieval) — reducing hallucination.
4. **Quantifies the stakes** with a deterministic economic-exposure estimate (`$ at risk` for a given field size).

It degrades gracefully: with no cloud credentials it falls back to a curated, citation-backed knowledge base and offline reasoning, so the demo **always runs**.

## 🧠 How it works

```
 leaf image
     │
     ▼
 PlantDoctor (predict.py)          ResNet18 + checkpoint → top-k disease classes
     │
     ▼
 ReasoningAgent (agent.py) ──────────────────────────────────────────────┐
     │  Step 1  Differential diagnosis  (parallel, multi-retrieval)        │
     │          • separate Foundry IQ retrieval per candidate              │
     │          • weigh distinguishing symptoms → verdict + what to check  │
     │  Step 2  Action plan: diagnosis · severity · treatment · economics  │
     │          • grounded + cited                                         │
     ▼                                                                     │
 FoundryIQ (foundry.py)  ◄───── live: Azure AI Search agentic retrieval ──┘
     │                          fallback: curated cited agronomy KB
     ▼
 economics.py → "$X–$Y at risk if untreated"  (deterministic estimate)
     │
     ▼
 Web app (api.py + website/)  /  Gradio app (app.py)
```

## ✨ Key features

- **Vision model** — ResNet18 transfer learning, **95.02% test accuracy** over 38 PlantVillage classes.
- **Differential-diagnosis reasoning loop** — genuine multi-step, tool-using reasoning that adjudicates competing diagnoses with grounded evidence (runs the retrievals in parallel for ~6s latency).
- **Microsoft Foundry IQ grounding** — answers are retrieved and cited from an Azure AI Search knowledge base built from the project's agronomy docs.
- **Economic exposure calculator** — a transparent, code-computed `$ at risk` estimate (not LLM-guessed), labeled as an estimate.
- **Confidence-aware & safe** — low-confidence calls are de-escalated and flagged; healthy leaves produce no false alarms; every claim carries a citation and an "consult your extension service" disclaimer.
- **Always demoable** — automatic offline fallback for both reasoning and grounding.

## 🚀 Quick start

```bash
pip install -r requirements.txt
```

**Model checkpoint** — the trained ResNet18 weights (`resnet18_best.pth`, ~129 MB) are too large for a normal Git repo, so they're distributed separately. Get them one of two ways:
- Download from the repo's [Releases](https://github.com/SNMiguel/PlantGuard-AI/releases) and place at `models/saved/resnet18_best.pth`, **or**
- Retrain: `python main.py --data_dir data/raw --epochs 10` (needs the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)).

```bash
# Single-image diagnosis (CLI)
python predict.py demo_images/1_tomato_late_blight.jpg

# Launch the full web app — marketing site + in-page leaf scanning (one server)
python api.py            # → http://127.0.0.1:8000

# Or the standalone Gradio demo (alternative UI)
python app.py            # → http://127.0.0.1:7860
```

### Optional: enable live LLM reasoning + Foundry IQ
Copy `.env.example` to `.env` and fill in credentials (see the file for options — GitHub Models / Azure OpenAI for reasoning, Azure AI Search for Foundry IQ). Everything works *without* these via offline fallbacks.

```bash
python foundry.py --selftest-live      # validate the live Foundry IQ wiring
```

To provision your own Foundry IQ knowledge base from the bundled agronomy docs, see `tools/export_kb.py` and `tools/setup_foundry_iq.py`.

## 📁 Project structure

```
PlantGuard-AI/
├── predict.py            # Vision inference (ResNet18 checkpoint) → top-k classes
├── agent.py              # Multi-step reasoning agent (differential + action plan)
├── foundry.py            # Microsoft Foundry IQ grounding (live + offline fallback)
├── economics.py          # Deterministic economic-exposure estimator
├── config.py             # Env/.env configuration for LLM + Foundry IQ
├── app.py                # Gradio demo UI (standalone/fallback)
├── api.py                # FastAPI backend: /api/diagnose + serves website/
├── main.py / train.py    # Training pipeline (transfer learning)
├── models/
│   ├── resnet_model.py   # ResNetClassifier
│   └── saved/            # Trained checkpoint (resnet18_best.pth)
├── utils/                # data_loader, transforms, metrics, visualizer
├── knowledge_base/       # 27 cited agronomy docs (indexed by Foundry IQ)
├── tools/
│   ├── export_kb.py          # Export the agronomy KB to upload-ready files
│   └── setup_foundry_iq.py   # Provision the Azure AI Search knowledge base
├── website/              # Single-page web app + marketing site (served by api.py)
├── demo_images/          # Curated sample leaves for the demo
├── DEMO_SCRIPT.md        # 60-second demo video script
└── SUBMISSION.md         # Hackathon submission notes
```

## 📊 Model performance

| Metric | Value |
|---|---|
| Test accuracy | 95.02% |
| Precision | 95.75% |
| Recall | 95.02% |
| F1-score | 95.01% |
| Classes / crops | 38 / 14 |

## 🌱 Supported crops & diseases

**Crops:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato — **38 disease + healthy classes** total.

## 🛠️ Technical notes

- **Vision:** ResNet18 (ImageNet pre-trained), 224×224 RGB, FC head replaced for 38 classes, 70/15/15 split, Adam + ReduceLROnPlateau.
- **Reasoning engine:** Azure AI Foundry / any OpenAI-compatible endpoint, with an offline knowledge-base composer as fallback.
- **Grounding:** Foundry IQ knowledge base backed by Azure AI Search agentic retrieval (`minimal` reasoning effort = no extra LLM cost), with a curated cited KB as fallback.
- **Platform:** `num_workers=0` throughout for Windows compatibility.

## 📄 License

MIT License.

## 👤 Author

**Miguel Shema Ngabonziza**
- LinkedIn: [linkedin.com/in/migztech](https://linkedin.com/in/migztech)
- GitHub: [github.com/SNMiguel](https://github.com/SNMiguel)
- Portfolio: [migztech.vercel.app](https://migztech.vercel.app)

## 🙏 Acknowledgments

- PlantVillage dataset (Penn State University).
- Agronomy knowledge grounded in public extension sources (UC IPM, Cornell, PennState, CABI, FAO, APS).
- Built for the **Microsoft Agents League @ AISF 2026** — Reasoning Agents track.
