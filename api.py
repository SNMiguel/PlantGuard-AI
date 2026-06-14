"""
PlantGuard-AI — FastAPI backend.

Wraps the existing inference + reasoning pipeline (predict.py, agent.py,
economics.py) behind a small JSON API, and serves the static website so the
front-end can render diagnoses in the site's own design — no Gradio iframe.

This does not replace app.py (the Gradio demo); it sits alongside it.

Run:
    python api.py            # -> http://127.0.0.1:8000
"""
import io
import sys

# Force UTF-8 console output (the underlying modules print Unicode glyphs).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from predict import get_doctor
from agent import reasoning_agent
import economics

# Human-friendly labels for the reasoning engine actually used.
ENGINE_LABELS = {
    "azure-openai": "Azure AI Foundry",
    "openai": "OpenAI-compatible LLM",
    "knowledge-base": "PlantGuard reasoning (offline knowledge base)",
}

DISCLAIMER = (
    "PlantGuard-AI provides decision support, not a substitute for professional "
    "agronomic or regulatory advice. Confirm chemical choices and rates with your "
    "local agricultural extension service before applying."
)

WEBSITE_DIR = Path(__file__).parent / "website"

app = FastAPI(title="PlantGuard-AI API", docs_url="/api/docs")


@app.get("/api/health")
async def health():
    """Lightweight readiness probe for the front-end."""
    return {"status": "ok"}


@app.post("/api/diagnose")
async def diagnose(image: UploadFile = File(...), acres: float = Form(10.0)):
    """Run the full vision + reasoning pipeline and return structured JSON.

    The front-end renders this with the site's own design language.
    """
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    # --- Vision model ---
    try:
        doctor = get_doctor()
        result = doctor.predict(pil, top_k=3)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model checkpoint not found: {exc}")
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed.")

    # --- Reasoning agent (grounded by Foundry IQ) ---
    report = reasoning_agent.run(
        label=result["label"],
        confidence=result["confidence"],
        top_k=result["top_k"],
        species=result["species"],
        condition=result["condition"],
    )

    # --- Economic exposure (deterministic; skipped when healthy) ---
    econ = None if result["is_healthy"] else economics.estimate(
        report.crop, report.severity, acres)

    payload = {
        "species": result["species"],
        "condition": result["condition"],
        "is_healthy": result["is_healthy"],
        "confidence": result["confidence"],
        "status": "Healthy" if result["is_healthy"] else "Disease detected",
        "severity": report.severity,
        "top_k": [
            {
                "species": p["species"],
                "condition": p["condition"],
                "confidence": p["confidence"],
            }
            for p in result["top_k"]
        ],
        "agent": {
            "headline": report.headline,
            "engine": ENGINE_LABELS.get(report.reasoning_engine, report.reasoning_engine),
            "grounding": (
                "Microsoft Foundry IQ"
                if report.grounding_source == "foundry-iq"
                else "Foundry IQ knowledge base"
            ),
            "differential": report.differential,  # dict or None (already JSON-safe)
            "steps": [
                {"icon": s.icon, "title": s.title, "content": s.content}
                for s in report.steps
            ],
            "citations": [
                {"url": c["url"], "title": c["title"], "source": c["source"]}
                for c in report.citations
            ],
        },
        "economics": econ,
        "disclaimer": DISCLAIMER,
    }
    return JSONResponse(payload)


# Serve the static site last so the /api/* routes take precedence.
app.mount("/", StaticFiles(directory=str(WEBSITE_DIR), html=True), name="site")


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  PlantGuard-AI server")
    print("  Site + API -> http://127.0.0.1:8000")
    print("  API docs   -> http://127.0.0.1:8000/api/docs")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000)
