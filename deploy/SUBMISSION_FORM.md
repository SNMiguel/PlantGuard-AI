# Innovation Studio submission: paste-ready field values

Copy each block into the matching field on the Agents League "Project" form.

---

## Challenge (dropdown)
**Reasoning Agents**

## Are you writing code as part of this project?
**Yes**

## Is this project open for others to join?
**No** (solo project; flip to Yes only if you want collaborators)

---

## Title  (max 140)
```
PlantGuard-AI: a reasoning agent that turns one leaf photo into a grounded, cited crop-disease action plan
```

## Tagline  (max 300)
```
Upload one crop-leaf photo and PlantGuard-AI runs a 95%-accurate vision model plus a reasoning agent that delivers a grounded differential diagnosis, the dollars at risk, and a cited treatment plan, every answer grounded by Microsoft Foundry IQ.
```

## Keywords  (add each as a tag)
```
reasoning agent
agentic retrieval
plant disease detection
agriculture
computer vision
ResNet18
PyTorch
Microsoft Foundry IQ
Azure AI Search
differential diagnosis
RAG
AgriTech
```

## Skills you are looking for  (optional; only if "open to join" = Yes)
```
Python
PyTorch
Azure AI
Frontend
```

---

## Description  (Markdown supported)
```
## PlantGuard-AI

A reasoning agent that turns a single crop-leaf photo into a grounded, cited action plan: diagnosis, severity, treatment, and economic impact.

### The problem
Plant disease costs farmers billions of dollars each year, and a diagnosis alone does not tell them what to do about it. Growers need to know how serious it is, what to apply, and what is at stake if they wait.

### What it does
1. A farmer uploads a leaf photo and field size in the PlantGuard web app, which runs the full diagnosis in-page.
2. A ResNet18 transfer-learning model classifies it across 38 disease classes and 14 crop species at 95% test accuracy, returning ranked candidates.
3. **Differential diagnosis (agentic):** when the call is ambiguous, the agent runs a separate Microsoft Foundry IQ retrieval per candidate, weighs the distinguishing symptoms, picks a verdict, and tells the farmer exactly what to inspect to confirm or rule out the runner-up. The retrievals run in parallel.
4. **Action plan:** Diagnosis, Severity, Treatment, and Economic Impact, with severity modulated by model confidence (low-confidence calls are de-escalated and flagged).
5. **Economic exposure:** a deterministic, code-computed dollars-at-risk estimate for the field, clearly labeled an estimate and never fabricated by the model.
6. Every claim is grounded with citations from authoritative agronomy sources (UC IPM, Cornell, PennState, CABI, FAO, APS) through a live Microsoft Foundry IQ knowledge base.

### How it uses the reasoning-agent and Microsoft stack
- **Reasoning agent:** genuine multi-step, multi-retrieval reasoning (differential diagnosis plus action plan) over the classifier output, the core of the Reasoning Agents track.
- **Microsoft Foundry IQ:** every retrieval is a live Azure AI Search agentic-retrieval call against a knowledge base built from the project's agronomy docs; the differential issues several per diagnosis.
- **Azure AI Foundry / OpenAI-compatible LLM:** the reasoning and adjudication engine.
- **Graceful degradation:** if cloud services are unavailable, the agent falls back to a curated, citation-backed knowledge base and offline reasoning, so it is always demoable.

### Results
95.02% test accuracy, 95.75% precision, 95.01% F1 on 8,147 held-out images across 38 classes.

### Run it
Clone the repo, install requirements, download the model checkpoint from the v1.0.0 Release into models/saved/, then run `python api.py` and open http://127.0.0.1:8000.

- Code: https://github.com/SNMiguel/PlantGuard-AI
- Demo video: https://www.youtube.com/watch?v=alXvoNP0Eg4
```

---

## After you click Save (fields that unlock)
- **Code Repository:** `https://github.com/SNMiguel/PlantGuard-AI`
- **Media:** upload a screenshot of the web app (the diagnosis result + agent reasoning) and/or the demo video
- **Demo video link:** https://www.youtube.com/watch?v=alXvoNP0Eg4
