# PlantGuard-AI — Submission Notes

Ready-to-paste content for the Microsoft Agents League submission form
(Reasoning Agents track). Edit to fit the form's exact fields/limits.

---

## Project name
PlantGuard-AI

## Tagline (one line)
A reasoning agent that turns a single crop-leaf photo into a grounded, cited action plan — diagnosis, severity, treatment, and economic impact.

## Track
Microsoft Agents League — Reasoning Agents

## Elevator pitch (~50 words)
Plant disease costs farmers billions each year, and a diagnosis alone doesn't tell them what to *do*. PlantGuard-AI pairs a 95%-accurate vision model with a multi-step reasoning agent that thinks through severity, treatment, and economic impact — every answer grounded and cited via Microsoft Foundry IQ.

## What it does
1. A farmer uploads a leaf photo in a Gradio web app.
2. A ResNet18 transfer-learning model classifies it across **38 disease classes / 14 crop species** at **95% test accuracy**.
3. A reasoning agent runs a **4-step chain** — Diagnosis → Severity → Treatment → Economic Impact — modulating severity by the model's confidence (low-confidence calls are de-escalated and flagged).
4. Answers are **grounded with citations** from authoritative agronomy sources (UC IPM, Cornell, PennState, CABI, FAO, APS) via a **Microsoft Foundry IQ** knowledge layer.

## How it uses the Microsoft / reasoning-agent stack
- **Reasoning agent:** multi-step agronomic reasoning over the classifier output (the core of the Reasoning Agents track).
- **Azure AI Foundry:** the LLM reasoning engine (OpenAI-compatible; runs on Azure AI Foundry model deployments).
- **Microsoft Foundry IQ:** grounds every answer with cited knowledge retrieved from an Azure AI Search–backed knowledge base (agentic retrieval).
- **Graceful degradation:** if cloud services are unavailable, the agent falls back to a curated, citation-backed knowledge base — so it is always demoable.

## Tech stack
PyTorch · ResNet18 (transfer learning) · Gradio · Azure AI Foundry · Microsoft Foundry IQ (Azure AI Search agentic retrieval) · Python

## Key metrics
- 95.02% test accuracy · 95.75% precision · 95.01% F1 · 8,147 test images · 38 classes

## Repo layout (for judges)
- `predict.py` — vision inference (ResNet18 checkpoint)
- `agent.py` — multi-step reasoning agent
- `foundry.py` — Foundry IQ grounding (live Azure AI Search + offline KB fallback)
- `app.py` — Gradio demo UI
- `website/` — marketing/demo site
- `knowledge_base/` — the 27 grounded agronomy docs indexed by Foundry IQ

## Links (fill in)
- Demo video: ____
- GitHub repo: ____
- Live demo (if hosted): ____
