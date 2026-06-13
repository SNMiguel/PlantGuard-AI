"""
Export PlantGuard-AI's curated agronomy knowledge base to upload-ready files.

Reads the citation-backed knowledge in foundry.py and writes one Markdown
document per disease into ./knowledge_base/. Upload that folder to an Azure
Blob Storage container, then point a Foundry IQ knowledge source at it.

Run:
    python tools/export_kb.py
"""
import re
import sys
from pathlib import Path

# Make the repo root importable when run as `python tools/export_kb.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foundry import _KB, _HEALTHY  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


def safe_name(label):
    """Turn a class label into a safe, readable filename stem."""
    name = label.replace("___", "__")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def render(label, entry):
    """Render one disease entry as a Markdown document."""
    readable = label.replace("___", " — ").replace("_", " ")
    lines = [f"# {readable}", ""]

    crop = label.split("___")[0].replace("_", " ")
    condition = ("Healthy" if label.endswith("healthy")
                 else label.split("___")[-1].replace("_", " "))
    lines += [
        f"**Crop:** {crop}  ",
        f"**Condition:** {condition}  ",
        f"**Pathogen:** {entry.get('pathogen', 'n/a')}  ",
        f"**Baseline severity if untreated:** {entry.get('severity', 'unknown')}",
        "",
        "## Summary",
        entry.get("summary", ""),
        "",
    ]

    passages = entry.get("passages", [])
    if passages:
        lines += ["## Key facts"]
        lines += [f"- {p}" for p in passages]
        lines += [""]

    lines += [
        "## Recommended treatment",
        entry.get("treatment", "Consult a local agricultural extension service."),
        "",
        "## Economic and crop impact",
        entry.get("economic", "Impact varies with severity and timing."),
        "",
        "## Sources",
    ]
    for c in entry.get("citations", []):
        lines.append(f"- {c.title} — {c.source} ({c.url})")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = dict(_KB)
    items["Apple___healthy_template"] = _HEALTHY  # one generic healthy doc

    count = 0
    for label, entry in items.items():
        stem = "_healthy_template" if label.endswith("_template") else safe_name(label)
        path = OUT_DIR / f"{stem}.md"
        path.write_text(render(label, entry), encoding="utf-8")
        count += 1

    print(f"Wrote {count} knowledge documents to {OUT_DIR}")
    print("Next: upload this folder to an Azure Blob Storage container,")
    print("then create a Foundry IQ knowledge source pointing at it.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
