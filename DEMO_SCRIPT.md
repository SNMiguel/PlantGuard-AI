# PlantGuard-AI — 60-Second Demo Script

**Track:** Microsoft Agents League — Reasoning Agents
**Goal:** Show a single leaf photo become a grounded, multi-step action plan.

## Before you hit record
1. Start the app: `venv\Scripts\python app.py` → open http://127.0.0.1:7860
2. Set **Field size = 20 acres** (powers the live economic-exposure number)
3. Open File Explorer to `C:\Users\smigu\PlantGuard-AI\demo_images\` (drag source)
4. Have the website open in a second tab: `website\index.html` (for the opening shot)
5. Close notifications / extra tabs. Record at 1080p. (Each diagnosis takes ~6s — that pause is the agent running its grounded retrievals; it reads as "thinking.")

---

## Shot list (≈60s)

| Time | On screen | Say (voiceover or captions) |
|------|-----------|------------------------------|
| **0:00–0:07** | Website hero (`index.html`) | "Plant disease costs farmers billions a year. PlantGuard-AI turns one leaf photo into an expert action plan." |
| **0:07–0:15** | Switch to the Gradio app | "It pairs a 95%-accurate vision model with a reasoning agent — grounded by Microsoft Foundry IQ." |
| **0:15–0:36** | Drag `4_tomato_blight_ambiguous.jpg` in → **Diagnose** → scroll to **🔎 Differential Diagnosis** | "Here the model is genuinely torn — early blight vs late blight. Watch the agent reason: it runs separate Foundry IQ retrievals on each candidate, weighs the distinguishing symptoms, picks early blight, and tells the farmer exactly what to inspect to confirm. That's multi-step reasoning, not a single guess." |
| **0:36–0:46** | Point to the **"💰 $16K–$32K at risk"** card, then the 4 action steps + "Microsoft Foundry IQ" sources | "It even quantifies the stakes — up to $32,000 at risk across 20 acres — then builds the action plan: severity, treatment, next steps, every claim grounded and cited live by Microsoft Foundry IQ from our own Azure knowledge base." |
| **0:46–0:53** | Drag `3_apple_healthy.jpg` in → Diagnose | "It also confirms healthy crops — no false alarms — across 38 diseases and 14 species." |
| **0:53–0:60** | Website features / footer | "PlantGuard-AI: a reasoning agent that protects the harvest. Vision, reasoning, and Foundry IQ grounding — built on Microsoft Azure." |

---

## Tips
- **Lead with the payoff.** Show a diagnosis result within the first 20 seconds.
- If the agent text takes a second to load, that's the LLM thinking — fine, it reads as "reasoning."
- Keep the cursor movements slow and deliberate; drag-drop reads better than file-picker.
- One clean take of the Tomato diagnosis is the heart of the video — nail that.

## Backup line (if Tier 2 / Foundry IQ is live by record time)
> "...grounded live by a Microsoft Foundry IQ knowledge base — see the cited sources pulled straight from our agricultural knowledge index."
