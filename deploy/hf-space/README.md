---
title: PlantGuard AI
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PlantGuard-AI

AI-powered plant disease detection and agronomic reasoning. Upload a crop leaf and a
95%-accurate vision model plus a reasoning agent return a diagnosis, severity, treatment,
economic exposure, and cited sources, grounded by Microsoft Foundry IQ.

This Space runs the FastAPI web app from the project repo. It builds by cloning
[SNMiguel/PlantGuard-AI](https://github.com/SNMiguel/PlantGuard-AI) and pulling the model
checkpoint from the v1.0.0 GitHub Release. With no cloud credentials set, the reasoning and
grounding fall back to a curated, citation-backed knowledge base, so the demo always works.

Source and full docs: https://github.com/SNMiguel/PlantGuard-AI
