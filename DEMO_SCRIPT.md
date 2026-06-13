# PlantGuard-AI — 60-Second Demo Script

**Track:** Microsoft Agents League — Reasoning Agents
**Goal:** Show a single leaf photo become a grounded, multi-step action plan.

## Before you hit record
1. Start the app: `venv\Scripts\python app.py` → open http://127.0.0.1:7860
2. Open File Explorer to `C:\Users\smigu\PlantGuard-AI\demo_images\` (drag source)
3. Have the website open in a second tab: `website\index.html` (for the opening shot)
4. Close notifications / extra tabs. Record at 1080p.

---

## Shot list (≈60s)

| Time | On screen | Say (voiceover or captions) |
|------|-----------|------------------------------|
| **0:00–0:07** | Website hero (`index.html`) | "Plant disease costs farmers billions a year. PlantGuard-AI turns one leaf photo into an expert action plan." |
| **0:07–0:15** | Switch to the Gradio app | "It pairs a 95%-accurate vision model with a reasoning agent — grounded by Microsoft Foundry IQ." |
| **0:15–0:33** | Drag `1_tomato_late_blight.jpg` in → click **Diagnose** → result appears | "Upload a leaf. The model detects Tomato late blight at 99.8% confidence — and the agent reasons step by step: diagnosis, severity, treatment, and economic impact." |
| **0:33–0:42** | Scroll to the agent reasoning steps + the "Microsoft Foundry IQ" sources | "Every recommendation is grounded live by Microsoft Foundry IQ — retrieved and cited from our own Azure AI Search knowledge base, not a hallucination." |
| **0:42–0:52** | Drag `3_apple_healthy.jpg` in → Diagnose | "It also confirms healthy crops — no false alarms — across 38 diseases and 14 species." |
| **0:52–0:60** | Website features / footer | "PlantGuard-AI: a reasoning agent that protects the harvest. Vision, reasoning, and Foundry IQ grounding — built on Microsoft Azure." |

---

## Tips
- **Lead with the payoff.** Show a diagnosis result within the first 20 seconds.
- If the agent text takes a second to load, that's the LLM thinking — fine, it reads as "reasoning."
- Keep the cursor movements slow and deliberate; drag-drop reads better than file-picker.
- One clean take of the Tomato diagnosis is the heart of the video — nail that.

## Backup line (if Tier 2 / Foundry IQ is live by record time)
> "...grounded live by a Microsoft Foundry IQ knowledge base — see the cited sources pulled straight from our agricultural knowledge index."
