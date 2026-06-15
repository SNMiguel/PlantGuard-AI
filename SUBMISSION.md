# PlantGuard-AI: Submission Notes

Ready-to-paste content for the Microsoft Agents League submission form
(Reasoning Agents track). Edit to fit the form's exact fields/limits.

---

## Project name
PlantGuard-AI

## Tagline (one line)
A reasoning agent that turns a single crop-leaf photo into a grounded, cited action plan: diagnosis, severity, treatment, and economic impact.

## Track
Microsoft Agents League, Reasoning Agents

## Elevator pitch (~50 words)
Plant disease costs farmers billions each year, and a diagnosis alone doesn't tell them what to *do*. PlantGuard-AI pairs a 95%-accurate vision model with a reasoning agent that runs a grounded **differential diagnosis**, quantifies the **dollars at risk**, and prescribes treatment, with every answer cited via Microsoft Foundry IQ.

## What it does
1. A farmer uploads a leaf photo (and field size) in the PlantGuard web app, which runs the full diagnosis in-page.
2. A ResNet18 transfer-learning model classifies it across **38 disease classes / 14 crop species** at **95% test accuracy**, returning ranked candidates.
3. **Step 1, differential diagnosis (agentic):** when the call is ambiguous, the agent runs a *separate Foundry IQ retrieval per candidate*, weighs the distinguishing symptoms, picks a verdict, and tells the farmer exactly what to inspect to confirm or rule out the runner-up. (Branches run in parallel.)
4. **Step 2, action plan:** Diagnosis → Severity → Treatment → Economic Impact, with severity modulated by model confidence (low-confidence calls are de-escalated and flagged).
5. **Economic exposure:** a deterministic, code-computed `$ at risk` estimate for the field, clearly labeled an estimate and never LLM-fabricated.
6. Every claim is **grounded with citations** from authoritative agronomy sources (UC IPM, Cornell, PennState, CABI, FAO, APS) via a live **Microsoft Foundry IQ** knowledge base.

## How it uses the Microsoft / reasoning-agent stack
- **Reasoning agent:** genuine multi-step, multi-retrieval reasoning (differential diagnosis + action plan) over the classifier output, the core of the Reasoning Agents track.
- **Microsoft Foundry IQ (required IQ layer):** every retrieval is a live Azure AI Search **agentic-retrieval** call against a knowledge base built from the project's agronomy docs; the differential issues several per diagnosis.
- **Azure AI Foundry / OpenAI-compatible LLM:** the reasoning/adjudication engine.
- **Graceful degradation:** if cloud services are unavailable, the agent falls back to a curated, citation-backed knowledge base and offline reasoning, so it is always demoable.

## Tech stack
PyTorch · ResNet18 (transfer learning) · FastAPI · Gradio · Azure AI Foundry · Microsoft Foundry IQ (Azure AI Search agentic retrieval) · Python

## Key metrics
- 95.02% test accuracy · 95.75% precision · 95.01% F1 · 8,147 test images · 38 classes

## Repo layout (for judges)
- `predict.py`: vision inference (ResNet18 checkpoint) → top-k classes
- `agent.py`: multi-step reasoning agent (differential diagnosis + action plan)
- `foundry.py`: Foundry IQ grounding (live Azure AI Search agentic retrieval + offline KB fallback)
- `economics.py`: deterministic economic-exposure estimator
- `api.py`: FastAPI backend serving `POST /api/diagnose` and the web app (`python api.py` → http://127.0.0.1:8000)
- `app.py`: Gradio UI (standalone/fallback)
- `website/`: the web app + marketing site (served by `api.py`); native in-page leaf scanning
- `knowledge_base/`: the 27 grounded agronomy docs indexed by Foundry IQ
- `tools/`: KB export + Foundry IQ provisioning scripts

## Links
- Demo video: https://www.youtube.com/watch?v=alXvoNP0Eg4
- GitHub repo: https://github.com/SNMiguel/PlantGuard-AI
- Live demo (if hosted): ____
