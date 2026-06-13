"""
PlantGuard-AI reasoning agent (Microsoft Agents League — Reasoning Agents track).

Takes the vision classifier's output (disease label + confidence) and performs
multi-step agronomic reasoning, grounded by Microsoft Foundry IQ:

    Step 1 — Diagnosis:        What is the disease / what was detected?
    Step 2 — Severity:         How severe is it (given confidence + pathogen)?
    Step 3 — Treatment:        What is the recommended treatment plan?
    Step 4 — Economic impact:  What is the potential crop / economic impact?

The agent calls an Azure AI Foundry model deployment (or any OpenAI-compatible
endpoint) for reasoning. When no LLM is configured it composes a structured
answer directly from the Foundry IQ grounded knowledge, so the agent is always
demoable. Every answer carries citations from the grounding layer.

Public API:
    agent = ReasoningAgent()
    report = agent.run(label="Tomato___Late_blight", confidence=0.99)
    # -> AgentReport(headline, steps[], citations[], grounded, ...)
"""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List

import requests

from config import config
from foundry import foundry_iq


@dataclass
class ReasoningStep:
    title: str
    icon: str
    content: str

    def to_dict(self):
        return {"title": self.title, "icon": self.icon, "content": self.content}


@dataclass
class AgentReport:
    label: str
    crop: str
    condition: str
    is_healthy: bool
    confidence: float
    headline: str
    severity: str
    steps: List[ReasoningStep] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)
    grounding_source: str = "local-kb"
    reasoning_engine: str = "knowledge-base"   # or "azure-openai" / "openai"
    grounded: bool = True
    differential: dict = None   # multi-step differential-diagnosis trace (or None)

    def to_dict(self):
        return {
            "label": self.label,
            "crop": self.crop,
            "condition": self.condition,
            "is_healthy": self.is_healthy,
            "confidence": self.confidence,
            "headline": self.headline,
            "severity": self.severity,
            "steps": [s.to_dict() for s in self.steps],
            "citations": self.citations,
            "grounding_source": self.grounding_source,
            "reasoning_engine": self.reasoning_engine,
            "grounded": self.grounded,
            "differential": self.differential,
        }


SEVERITY_RANK = {"none": 0, "low": 1, "moderate": 2, "high": 3, "severe": 4,
                 "unknown": 2}

STEP_META = [
    ("Diagnosis", "🔬"),
    ("Severity Assessment", "📊"),
    ("Recommended Treatment", "💊"),
    ("Economic & Crop Impact", "📉"),
]

SYSTEM_PROMPT = (
    "You are PlantGuard, an expert agronomist and plant pathologist AI agent. "
    "You receive a plant disease diagnosis from a vision model plus grounded "
    "knowledge retrieved from authoritative agricultural sources. Reason step by "
    "step and produce a clear, actionable report for a farmer. "
    "Base every claim ONLY on the grounded knowledge provided; do not invent "
    "chemicals, statistics, or sources. Be concise, practical, and confident. "
    "Respond ONLY with valid JSON of the form: "
    '{"headline": str, "severity": one of '
    '["none","low","moderate","high","severe"], '
    '"diagnosis": str, "severity_explanation": str, "treatment": str, '
    '"economic_impact": str}. '
    "Each text field should be 2-4 sentences."
)


class ReasoningAgent:
    """Multi-step reasoning agent grounded by Foundry IQ."""

    def __init__(self):
        self.foundry = foundry_iq

    def run(self, label, confidence, top_k=None, species=None, condition=None):
        """Produce a grounded, multi-step reasoning report.

        Args:
            label: raw class label, e.g. "Tomato___Late_blight".
            confidence: classifier confidence (0-1).
            top_k: optional ranked predictions from the classifier
                   (list of {label, species, condition, confidence}). When given,
                   the agent runs a differential-diagnosis step first.
            species/condition: optional pre-split readable parts.
        """
        is_healthy = label.endswith("healthy")

        crop = species or label.split("___")[0].replace("_", " ")
        cond = condition or (
            "Healthy" if is_healthy
            else label.split("___")[-1].replace("_", " ")
        )

        # The differential branch (Step 1) and the action-plan branch (Step 2)
        # are independent, so run them concurrently to cut latency roughly in half.
        if top_k and not is_healthy:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_diff = ex.submit(self._differential, top_k, confidence)
                fut_main = ex.submit(self._ground_and_reason,
                                     label, crop, cond, confidence, is_healthy)
                differential = fut_diff.result()
                main = fut_main.result()
        else:
            differential = None
            main = self._ground_and_reason(label, crop, cond, confidence, is_healthy)

        return self._build_report(
            label, crop, cond, is_healthy, confidence, main["payload"],
            main["citations"], main["grounding_source"], main["engine"], differential
        )

    # --- Step 2: ground the confirmed disease and reason -------------------
    def _ground_and_reason(self, label, crop, cond, confidence, is_healthy):
        """Ground the confirmed disease and produce the 4-section action plan."""
        grounding = self.foundry.ground(label)
        entry = self.foundry.get_entry(label)
        citations = [c.to_dict() for c in grounding.citations]

        if config.llm_configured:
            try:
                payload = self._reason_with_llm(
                    crop, cond, confidence, grounding, entry, is_healthy
                )
                engine = "azure-openai" if config.is_azure else "openai"
                return {"payload": payload, "citations": citations,
                        "grounding_source": grounding.source, "engine": engine}
            except Exception as exc:
                print(f"[agent] LLM reasoning failed, using knowledge base: {exc}")

        payload = self._reason_with_kb(crop, cond, confidence, entry, is_healthy)
        return {"payload": payload, "citations": citations,
                "grounding_source": grounding.source, "engine": "knowledge-base"}

    # --- Step 1: differential diagnosis (multi-step, multi-retrieval) -------
    def _differential(self, top_k, confidence):
        """Compare the top disease candidates using per-candidate Foundry IQ
        retrievals, then adjudicate which fits and what to check to confirm.

        Returns a dict (the visible reasoning trace) or None if there aren't
        at least two disease candidates to weigh.
        """
        candidates = [c for c in top_k if not c["label"].endswith("healthy")][:2]
        if len(candidates) < 2:
            return None

        # Separate, targeted retrieval per candidate (the agentic part), run
        # concurrently since the candidate retrievals are independent.
        def _retrieve(c):
            return c, self.foundry.ground(
                c["label"],
                query=(f"distinguishing symptoms and how to identify "
                       f"{c['condition']} versus similar diseases"),
            )

        with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
            results = list(ex.map(_retrieve, candidates))

        enriched, extra_citations = [], []
        for c, g in results:
            feature = g.passages[0] if g.passages else ""
            enriched.append({
                "crop": c["species"],
                "condition": c["condition"],
                "confidence": round(float(c["confidence"]), 4),
                "distinguishing": feature,
            })
            extra_citations.extend(cit.to_dict() for cit in g.citations[:2])

        margin = candidates[0]["confidence"] - candidates[1]["confidence"]

        if config.llm_configured:
            try:
                verdict = self._adjudicate_with_llm(enriched, margin, confidence)
            except Exception as exc:
                print(f"[agent] differential LLM failed, using heuristic: {exc}")
                verdict = self._adjudicate_with_kb(enriched, margin)
        else:
            verdict = self._adjudicate_with_kb(enriched, margin)

        # Dedup citations by title.
        seen, cites = set(), []
        for cit in extra_citations:
            if cit["title"] in seen:
                continue
            seen.add(cit["title"])
            cites.append(cit)

        return {
            "candidates": enriched,
            "verdict": verdict["verdict"],
            "rationale": verdict["rationale"],
            "confirm_checks": verdict["confirm_checks"],
            "retrievals": len(enriched),
            "margin": round(float(margin), 4),
            "citations": cites,
        }

    def _adjudicate_with_llm(self, enriched, margin, confidence):
        """Ask the LLM to weigh the candidates and decide."""
        cand_block = "\n".join(
            f"- {c['condition']} on {c['crop']} (model confidence "
            f"{c['confidence']*100:.1f}%). Distinguishing features: "
            f"{c['distinguishing'][:400]}"
            for c in enriched
        )
        prompt = (
            "You are differentiating between competing plant-disease diagnoses. "
            "Using ONLY the grounded distinguishing features and the vision "
            "model's confidences below, decide which diagnosis is most likely "
            "and what a farmer should physically check to confirm it or rule out "
            "the runner-up.\n\n"
            f"CANDIDATES:\n{cand_block}\n\n"
            f"Confidence margin between #1 and #2: {margin*100:.1f} points.\n\n"
            'Respond ONLY with JSON: {"verdict": str (the chosen condition), '
            '"rationale": str (2-3 sentences weighing the candidates), '
            '"confirm_checks": str (1-2 concrete things to inspect on the plant)}.'
        )
        messages = [
            {"role": "system", "content": "You are PlantGuard, a careful plant pathologist."},
            {"role": "user", "content": prompt},
        ]
        data = self._parse_json(self._call_llm(messages))
        kb = self._adjudicate_with_kb(enriched, margin)
        for k in ("verdict", "rationale", "confirm_checks"):
            if not data.get(k):
                data[k] = kb[k]
        return data

    @staticmethod
    def _adjudicate_with_kb(enriched, margin):
        """Heuristic adjudication when no LLM is configured."""
        c0, c1 = enriched[0], enriched[1]
        if margin >= 0.4:
            rationale = (
                f"{c0['condition']} is strongly favored "
                f"({c0['confidence']*100:.1f}%) over {c1['condition']} "
                f"({c1['confidence']*100:.1f}%); the visual signature clearly "
                f"matches {c0['condition']}.")
        else:
            rationale = (
                f"{c0['condition']} ({c0['confidence']*100:.1f}%) leads "
                f"{c1['condition']} ({c1['confidence']*100:.1f}%), but the margin "
                f"is narrow — {c1['condition']} can't be excluded from the image "
                f"alone and warrants a closer look.")
        c0_feat = (c0["distinguishing"].split(".")[0] or c0["condition"]).strip()
        c1_feat = (c1["distinguishing"].split(".")[0] or c1["condition"]).strip()
        confirm = (f"To confirm {c0['condition']}: {c0_feat}. "
                   f"To rule out {c1['condition']}: {c1_feat}.")
        return {"verdict": c0["condition"], "rationale": rationale,
                "confirm_checks": confirm}

    # --- Report assembly ----------------------------------------------------
    def _build_report(self, label, crop, cond, is_healthy, confidence, payload,
                      citations, grounding_source, engine, differential=None):
        contents = [
            payload["diagnosis"],
            payload["severity_explanation"],
            payload["treatment"],
            payload["economic_impact"],
        ]
        steps = [
            ReasoningStep(title=STEP_META[i][0], icon=STEP_META[i][1],
                          content=contents[i])
            for i in range(4)
        ]

        # Merge differential citations into the report (deduped by title).
        if differential and differential.get("citations"):
            seen = {c["title"] for c in citations}
            for cit in differential["citations"]:
                if cit["title"] not in seen:
                    seen.add(cit["title"])
                    citations.append(cit)

        return AgentReport(
            label=label,
            crop=crop,
            condition=cond,
            is_healthy=is_healthy,
            confidence=round(float(confidence), 4),
            headline=payload["headline"],
            severity=payload["severity"],
            steps=steps,
            citations=citations,
            grounding_source=grounding_source,
            reasoning_engine=engine,
            grounded=True,
            differential=differential,
        )

    # --- LLM reasoning ------------------------------------------------------
    def _reason_with_llm(self, crop, cond, confidence, grounding, entry, is_healthy):
        user_prompt = (
            f"VISION MODEL DIAGNOSIS:\n"
            f"- Crop: {crop}\n"
            f"- Detected condition: {cond}\n"
            f"- Classifier confidence: {confidence * 100:.1f}%\n"
            f"- Status: {'HEALTHY' if is_healthy else 'DISEASE DETECTED'}\n"
            f"- Likely pathogen: {entry.get('pathogen', 'n/a')}\n"
            f"- Baseline severity if untreated: {entry.get('severity', 'unknown')}\n\n"
            f"{grounding.as_context()}\n\n"
            f"Now reason step by step and produce the JSON report. Factor the "
            f"classifier confidence into your severity assessment."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = self._call_llm(messages)
        data = self._parse_json(raw)

        # Validate required keys, fill any gaps from the KB fallback.
        kb = self._reason_with_kb(crop, cond, confidence, entry, is_healthy)
        for key in ("headline", "severity", "diagnosis", "severity_explanation",
                    "treatment", "economic_impact"):
            if not data.get(key):
                data[key] = kb[key]
        return data

    def _call_llm(self, messages):
        """Call Azure OpenAI or an OpenAI-compatible chat completions endpoint."""
        body = {
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 700,
        }

        if config.is_azure:
            url = (
                f"{config.llm_endpoint.rstrip('/')}/openai/deployments/"
                f"{config.llm_deployment}/chat/completions"
                f"?api-version={config.azure_api_version}"
            )
            headers = {"api-key": config.llm_api_key,
                       "Content-Type": "application/json"}
        else:
            base = config.llm_endpoint.rstrip("/")
            if not base.endswith("/chat/completions"):
                base = base + "/chat/completions"
            url = base
            headers = {"Authorization": f"Bearer {config.llm_api_key}",
                       "Content-Type": "application/json"}
            body["model"] = config.llm_deployment

        resp = requests.post(url, json=body, headers=headers,
                             timeout=config.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_json(raw):
        """Extract a JSON object from a model response (tolerant of code fences)."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.lstrip().lower().startswith("json"):
                raw = raw.lstrip()[4:]
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
        return json.loads(raw)

    # --- Knowledge-base reasoning (offline fallback / default) --------------
    def _reason_with_kb(self, crop, cond, confidence, entry, is_healthy):
        pct = confidence * 100
        pathogen = entry.get("pathogen", "Unknown pathogen")
        base_summary = entry.get("summary", f"{cond} detected on {crop}.")
        treatment = entry.get("treatment", "Consult a local extension service.")
        economic = entry.get("economic", "Impact varies with severity and timing.")
        base_severity = entry.get("severity", "unknown")

        # Confidence-based qualifier on the diagnosis.
        if confidence >= 0.85:
            conf_phrase = f"high confidence ({pct:.1f}%)"
        elif confidence >= 0.6:
            conf_phrase = f"moderate confidence ({pct:.1f}%)"
        else:
            conf_phrase = (f"low confidence ({pct:.1f}%) — consider re-imaging the "
                           f"leaf in good light or seeking expert confirmation")

        if is_healthy:
            headline = f"{crop} leaf looks healthy"
            diagnosis = (
                f"The vision model classified this {crop} leaf as healthy with "
                f"{conf_phrase}. {base_summary}"
            )
            severity_explanation = (
                "No disease symptoms detected, so there is no severity concern at "
                "this time. Continue routine scouting to catch any early changes."
            )
            return {
                "headline": headline,
                "severity": "none",
                "diagnosis": diagnosis,
                "severity_explanation": severity_explanation,
                "treatment": treatment,
                "economic_impact": economic,
            }

        # Severity wording, modulated by both pathogen baseline and confidence.
        rank = SEVERITY_RANK.get(base_severity, 2)
        if confidence < 0.6 and rank > 1:
            rank -= 1  # de-escalate when the model is unsure
        severity_word = {0: "none", 1: "low", 2: "moderate", 3: "high",
                         4: "severe"}[rank]
        urgency = {
            "low": "Monitor and treat if it progresses.",
            "moderate": "Begin management soon to prevent spread.",
            "high": "Act promptly — this disease can spread and cause real loss.",
            "severe": "Act immediately — this is a high-impact disease that can "
                      "cause rapid, major crop loss.",
        }.get(severity_word, "Begin management soon.")

        headline = f"{cond} detected on {crop}"
        diagnosis = (
            f"The vision model detected {cond} on {crop} with {conf_phrase}. "
            f"Likely cause: {pathogen}. {base_summary}"
        )
        severity_explanation = (
            f"Estimated severity: {severity_word.upper()}. {urgency} This rating "
            f"reflects the disease's biology and the {pct:.0f}% model confidence."
        )
        return {
            "headline": headline,
            "severity": severity_word,
            "diagnosis": diagnosis,
            "severity_explanation": severity_explanation,
            "treatment": treatment,
            "economic_impact": economic,
        }


# Singleton
reasoning_agent = ReasoningAgent()


def analyze(label, confidence, top_k=None, species=None, condition=None):
    """Convenience wrapper returning a plain dict report."""
    return reasoning_agent.run(
        label, confidence, top_k=top_k, species=species, condition=condition
    ).to_dict()


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("Testing ReasoningAgent...\n")
    print(f"LLM configured: {config.llm_configured} "
          f"(engine will be {'LLM' if config.llm_configured else 'knowledge-base'})\n")

    # Synthetic top-k (as predict.py would return) to exercise the differential.
    cases = [
        ("Tomato___Late_blight", 0.9979, [
            {"label": "Tomato___Late_blight", "species": "Tomato",
             "condition": "Late blight", "confidence": 0.62},
            {"label": "Tomato___Early_blight", "species": "Tomato",
             "condition": "Early blight", "confidence": 0.31},
        ]),
        ("Apple___healthy", 0.97, None),
    ]
    for label, conf, top_k in cases:
        report = reasoning_agent.run(label, conf, top_k=top_k)
        print("=" * 72)
        print(f"{report.headline}   [severity: {report.severity.upper()} | "
              f"engine: {report.reasoning_engine} | grounding: {report.grounding_source}]")
        print("-" * 72)
        if report.differential:
            d = report.differential
            print(f"\n🔎 Differential Diagnosis  ({d['retrievals']} grounded retrievals)")
            for c in d["candidates"]:
                print(f"   • {c['condition']} ({c['confidence']*100:.1f}%)")
            print(f"   Verdict: {d['verdict']}")
            print(f"   {d['rationale']}")
            print(f"   Confirm: {d['confirm_checks']}")
        for step in report.steps:
            print(f"\n{step.icon} {step.title}")
            print(f"   {step.content}")
        print("\nCitations:")
        for c in report.citations:
            print(f"   • {c['title']} — {c['source']}")
        print()
    print("✓ Reasoning agent ready!")
