"""
PlantGuard-AI — Gradio demo app.

A farmer uploads a crop leaf image and gets:
  1. Disease detection result + confidence (ResNet vision model, predict.py)
  2. A multi-step AI agent response — diagnosis, severity, treatment, economic
     impact — grounded with cited agricultural knowledge (agent.py + foundry.py).

Run:
    python app.py
Then open the local URL Gradio prints (default http://127.0.0.1:7860).
"""
import sys
import traceback

# Force UTF-8 console output (the underlying modules print Unicode glyphs).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import gradio as gr

from predict import get_doctor
from agent import reasoning_agent
from config import config
import economics

# Brand palette (matches the website).
FOREST = "#1a4731"
MINT = "#86efac"

CUSTOM_CSS = """
.gradio-container {max-width: 1100px !important; margin: auto !important;}
#pg-header {text-align:center; padding: 18px 0 4px 0;}
#pg-header h1 {color:#1a4731; font-size: 2.2rem; margin-bottom: 2px;}
#pg-header p {color:#3f6f57; font-size: 1.02rem; margin-top: 0;}
.pg-badge {display:inline-block; padding:4px 14px; border-radius:999px;
  font-weight:700; font-size:0.85rem; letter-spacing:.3px; color:#fff;}
.sev-none   {background:#16a34a;}
.sev-low    {background:#65a30d;}
.sev-moderate {background:#ca8a04;}
.sev-high   {background:#ea580c;}
.sev-severe {background:#dc2626;}
.pg-card {border:1px solid #d6e7dc; border-radius:14px; padding:18px 20px;
  background:#f7fdf9; box-shadow:0 2px 10px rgba(26,71,49,.06);}
.pg-step {border-left:4px solid #86efac; padding:6px 0 6px 14px; margin:14px 0;}
.pg-step h4 {margin:0 0 4px 0; color:#1a4731;}
footer {visibility: hidden;}
"""

SEVERITY_CLASS = {
    "none": "sev-none", "low": "sev-low", "moderate": "sev-moderate",
    "high": "sev-high", "severe": "sev-severe", "unknown": "sev-moderate",
}

EXAMPLE_DISCLAIMER = (
    "PlantGuard-AI provides decision support, not a substitute for professional "
    "agronomic or regulatory advice. Confirm chemical choices and rates with your "
    "local agricultural extension service before applying."
)


def _confidence_bar(pct):
    """Return an HTML confidence meter."""
    color = MINT if pct >= 60 else "#fca5a5"
    return (
        f'<div style="background:#e5efe9;border-radius:999px;height:14px;'
        f'width:100%;overflow:hidden;margin:6px 0 2px 0;">'
        f'<div style="height:100%;width:{pct:.1f}%;background:{color};"></div></div>'
    )


def diagnose(image, acres=10):
    """Main inference + reasoning pipeline for the UI."""
    if image is None:
        return ("<div class='pg-card'>Please upload a leaf image to begin.</div>",
                "", "")

    try:
        doctor = get_doctor()
        result = doctor.predict(image, top_k=3)
    except FileNotFoundError as exc:
        return (f"<div class='pg-card'><b>Model not found.</b><br>{exc}</div>", "", "")
    except Exception:
        return (f"<div class='pg-card'><b>Prediction failed.</b><pre>"
                f"{traceback.format_exc()}</pre></div>", "", "")

    # --- Reasoning agent (grounded by Foundry IQ) ---
    report = reasoning_agent.run(
        label=result["label"],
        confidence=result["confidence"],
        top_k=result["top_k"],
        species=result["species"],
        condition=result["condition"],
    )

    pct = result["confidence"] * 100
    status = "Healthy" if result["is_healthy"] else "Disease detected"
    status_color = "#16a34a" if result["is_healthy"] else "#dc2626"
    sev_class = SEVERITY_CLASS.get(report.severity, "sev-moderate")

    # --- Result card ---
    result_html = f"""
    <div class='pg-card'>
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:.8rem;color:#6b8a7a;text-transform:uppercase;letter-spacing:.5px;">Crop</div>
          <div style="font-size:1.5rem;font-weight:800;color:#1a4731;">{result['species']}</div>
        </div>
        <span class="pg-badge {sev_class}">Severity: {report.severity.upper()}</span>
      </div>
      <div style="margin-top:10px;font-size:1.15rem;">
        <span style="color:{status_color};font-weight:700;">● {status}</span>
        &nbsp;—&nbsp;<b>{result['condition']}</b>
      </div>
      {_confidence_bar(pct)}
      <div style="font-size:.92rem;color:#3f6f57;">Model confidence: <b>{pct:.2f}%</b></div>
      <div style="margin-top:12px;padding-top:10px;border-top:1px dashed #cfe3d7;">
        <div style="font-size:.8rem;color:#6b8a7a;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Other possibilities</div>
        {"".join(
            f'<div style="font-size:.9rem;color:#4b5563;">{p["species"]} — {p["condition"]} '
            f'<span style="color:#9aa;">({p["confidence"]*100:.1f}%)</span></div>'
            for p in result["top_k"][1:]
        )}
      </div>
    </div>
    """

    # --- Economic exposure card (deterministic estimate) ---
    econ = None if result["is_healthy"] else economics.estimate(
        report.crop, report.severity, acres)
    if econ:
        result_html += f"""
    <div class='pg-card' style="margin-top:14px;background:linear-gradient(160deg,#0c2616,#1a4731) !important;border:1px solid #1a4731;">
      <div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.5px;color:#86efac !important;">💰 Estimated economic exposure</div>
      <div style="font-size:1.6rem;font-weight:800;margin:4px 0;color:#ffffff !important;">{econ['headline']}</div>
      <div style="font-size:.92rem;color:#dcefe2 !important;">{econ['subline']}</div>
      <div style="font-size:.72rem;color:#a8c7b6 !important;margin-top:8px;border-top:1px dashed rgba(255,255,255,.22);padding-top:6px;">{econ['disclaimer']}</div>
    </div>
    """

    # --- Reasoning steps ---
    engine_label = {
        "azure-openai": "Azure AI Foundry",
        "openai": "OpenAI-compatible LLM",
        "knowledge-base": "PlantGuard reasoning (offline knowledge base)",
    }.get(report.reasoning_engine, report.reasoning_engine)

    steps_html = (
        f"<div class='pg-card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<h3 style='margin:0;color:#1a4731;'>🧠 Agent reasoning</h3>"
        f"<span style='font-size:.78rem;color:#6b8a7a;'>via {engine_label}</span></div>"
        f"<p style='color:#3f6f57;margin:6px 0 4px 0;font-style:italic;'>{report.headline}</p>"
    )

    # --- Differential diagnosis (multi-step, multi-retrieval) ---
    if report.differential:
        d = report.differential
        cand_rows = "".join(
            f'<div style="display:flex;justify-content:space-between;font-size:.9rem;'
            f'padding:3px 0;"><span>{c["condition"]}</span>'
            f'<span style="color:#6b8a7a;">{c["confidence"]*100:.1f}%</span></div>'
            for c in d["candidates"]
        )
        steps_html += (
            f"<div class='pg-step' style='border-left-color:#1a4731;'>"
            f"<h4>🔎 Differential Diagnosis "
            f"<span style='font-weight:400;font-size:.78rem;color:#6b8a7a;'>"
            f"· {d['retrievals']} grounded retrievals</span></h4>"
            f"<div style='background:#eef7f1;border-radius:8px;padding:8px 12px;margin:6px 0;'>{cand_rows}</div>"
            f"<div style='color:#374151;'><b>Verdict:</b> {d['verdict']}. {d['rationale']}</div>"
            f"<div style='color:#1a4731;margin-top:6px;font-size:.92rem;'>"
            f"✔️ <b>To confirm:</b> {d['confirm_checks']}</div></div>"
        )

    for step in report.steps:
        steps_html += (
            f"<div class='pg-step'><h4>{step.icon} {step.title}</h4>"
            f"<div style='color:#374151;'>{step.content}</div></div>"
        )
    steps_html += "</div>"

    # --- Citations / grounding ---
    grounding_label = ("Microsoft Foundry IQ"
                       if report.grounding_source == "foundry-iq"
                       else "Foundry IQ knowledge base")
    cites = "".join(
        f'<li><a href="{c["url"]}" target="_blank" style="color:#1a4731;">'
        f'{c["title"]}</a> — {c["source"]}</li>'
        for c in report.citations
    )
    sources_html = f"""
    <div class='pg-card'>
      <h4 style="margin:0 0 6px 0;color:#1a4731;">📚 Grounded sources <span style="font-weight:400;font-size:.8rem;color:#6b8a7a;">({grounding_label})</span></h4>
      <ul style="margin:6px 0 8px 18px;color:#374151;">{cites}</ul>
      <div style="font-size:.78rem;color:#8aa;border-top:1px dashed #cfe3d7;padding-top:8px;">{EXAMPLE_DISCLAIMER}</div>
    </div>
    """

    return result_html, steps_html, sources_html


def build_demo():
    with gr.Blocks(css=CUSTOM_CSS, title="PlantGuard-AI",
                   theme=gr.themes.Soft(primary_hue="green")) as demo:
        gr.HTML(
            "<div id='pg-header'><h1>🌿 PlantGuard-AI</h1>"
            "<p>Upload a crop leaf — get an instant diagnosis and a grounded, "
            "multi-step treatment plan from the reasoning agent.</p></div>"
        )

        engine_note = ("Azure AI Foundry model" if config.llm_configured
                       else "offline knowledge base (set Azure/OpenAI keys to enable LLM reasoning)")
        gr.Markdown(
            f"<div style='text-align:center;color:#6b8a7a;font-size:.85rem;'>"
            f"Vision: ResNet18 · 38 classes · 95% accuracy &nbsp;|&nbsp; "
            f"Reasoning engine: {engine_note} &nbsp;|&nbsp; Grounding: Foundry IQ</div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Crop leaf image",
                                    height=320)
                acres_in = gr.Number(value=10, label="Field size (acres)",
                                     precision=0, minimum=0)
                btn = gr.Button("🔍 Diagnose", variant="primary", size="lg")
                gr.Markdown(
                    "<span style='font-size:.82rem;color:#6b8a7a;'>Tip: a clear, "
                    "well-lit photo of a single leaf works best. Field size powers "
                    "the economic exposure estimate.</span>"
                )
            with gr.Column(scale=1):
                result_out = gr.HTML()

        steps_out = gr.HTML()
        sources_out = gr.HTML()

        btn.click(fn=diagnose, inputs=[image_in, acres_in],
                  outputs=[result_out, steps_out, sources_out])
        image_in.upload(fn=diagnose, inputs=[image_in, acres_in],
                        outputs=[result_out, steps_out, sources_out])

    return demo


if __name__ == "__main__":
    print("Launching PlantGuard-AI demo...")
    print(f"  LLM reasoning configured: {config.llm_configured}")
    print(f"  Foundry IQ live retrieval: {config.foundry_configured}")
    demo = build_demo()
    demo.launch()
