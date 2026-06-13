"""
Microsoft Foundry IQ integration for PlantGuard-AI.

Foundry IQ is the "Microsoft IQ" knowledge layer for the Reasoning Agents track:
it grounds the agent's answers in authoritative agricultural sources and returns
*cited* passages so every recommendation is traceable to a reference.

Two modes:
  1. Live Foundry IQ retrieval — Foundry IQ knowledge bases are backed by Azure
     AI Search agentic retrieval. When the Search endpoint + key + knowledge base
     name are configured, we call the knowledge base "retrieve" action, which
     plans the query, runs hybrid/semantic search across the knowledge base's
     sources, and returns grounded chunks with references for citation.
     Docs: https://learn.microsoft.com/azure/search/agentic-retrieval-how-to-retrieve
  2. Curated offline knowledge base — otherwise we ground answers against a
     built-in, citation-backed agronomy KB drawn from university extension
     services, CABI, and FAO. This keeps the agent grounded (and demoable)
     with zero external dependencies.

Cost note: "minimal" reasoning effort uses no LLM (free-tier friendly); "low"/
"medium" add Azure OpenAI query-planning charges. See .env.example.

Public API:
    kb = FoundryIQ()
    grounding = kb.ground(disease_label, query="treatment and severity")
    # -> GroundedKnowledge(passages, citations, source, grounded)
"""
import json
from dataclasses import dataclass, field
from typing import List

import requests

from config import config


def _pretty_source_title(raw):
    """Turn a Foundry IQ source filename (e.g. 'Tomato_Late_blight.md') into a
    readable citation title (e.g. 'Tomato — Late blight')."""
    if not raw:
        return "Foundry IQ knowledge source"
    name = str(raw).rsplit("/", 1)[-1]          # drop any path
    name = name.rsplit(".", 1)[0]               # drop extension
    name = name.replace("__", " — ").replace("_", " ").strip()
    return name or "Foundry IQ knowledge source"


@dataclass
class Citation:
    title: str
    source: str
    url: str

    def to_dict(self):
        return {"title": self.title, "source": self.source, "url": self.url}


@dataclass
class GroundedKnowledge:
    disease: str
    summary: str
    passages: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    source: str = "local-kb"        # "foundry-iq" when live retrieval is used
    grounded: bool = True

    def to_dict(self):
        return {
            "disease": self.disease,
            "summary": self.summary,
            "passages": self.passages,
            "citations": [c.to_dict() for c in self.citations],
            "source": self.source,
            "grounded": self.grounded,
        }

    def as_context(self):
        """Render grounded knowledge as a context block for the LLM prompt."""
        lines = [f"GROUNDED KNOWLEDGE for {self.disease} (source: {self.source}):"]
        for i, passage in enumerate(self.passages, 1):
            lines.append(f"[{i}] {passage}")
        if self.citations:
            lines.append("CITATIONS:")
            for i, c in enumerate(self.citations, 1):
                lines.append(f"[{i}] {c.title} — {c.source} ({c.url})")
        return "\n".join(lines)


# --- Reusable citations to authoritative agricultural knowledge sources ---
_C = {
    "ipm": Citation("UC IPM Pest Management Guidelines",
                    "University of California Agriculture & Natural Resources",
                    "https://ipm.ucanr.edu/"),
    "cornell": Citation("Vegetable & Fruit Disease Fact Sheets",
                        "Cornell University Plant Pathology",
                        "https://www.vegetables.cornell.edu/"),
    "psu": Citation("Plant Disease Management",
                    "PennState Extension",
                    "https://extension.psu.edu/plant-disease-management"),
    "cabi": Citation("CABI Crop Protection Compendium",
                     "CABI",
                     "https://www.cabidigitallibrary.org/product/qc"),
    "fao": Citation("Plant Production and Protection",
                    "Food and Agriculture Organization (FAO)",
                    "https://www.fao.org/plant-production-protection/en"),
    "apsnet": Citation("Disease Lessons & Common Names of Plant Diseases",
                       "The American Phytopathological Society (APS)",
                       "https://www.apsnet.org/edcenter/disandpath/"),
}


# Curated, citation-backed knowledge keyed by raw class label.
# severity: typical baseline severity if untreated (low | moderate | high | severe)
# Each entry feeds both grounding (passages/citations) and the agent's reasoning.
_KB = {
    "Apple___Apple_scab": {
        "severity": "moderate",
        "pathogen": "Fungus (Venturia inaequalis)",
        "summary": "Apple scab is a fungal disease producing olive-green to black lesions on leaves and fruit, reducing fruit quality and causing early defoliation in wet springs.",
        "passages": [
            "Apple scab is caused by the fungus Venturia inaequalis and is favored by cool, wet spring weather; primary infections come from spores released from overwintered leaf litter.",
            "Management combines sanitation (removing fallen leaves), resistant cultivars, and protectant/systemic fungicide sprays timed to infection periods.",
        ],
        "treatment": "Apply protectant fungicides (e.g. captan or mancozeb) on a schedule during the primary infection window; rake and destroy fallen leaves to reduce overwintering inoculum.",
        "economic": "Unmanaged scab can render a large share of fruit unmarketable and trigger early defoliation that weakens trees over successive seasons.",
        "citations": [_C["ipm"], _C["apsnet"]],
    },
    "Apple___Black_rot": {
        "severity": "moderate",
        "pathogen": "Fungus (Botryosphaeria obtusa)",
        "summary": "Black rot causes leaf 'frogeye' spots, cankers on limbs, and a firm brown fruit rot.",
        "passages": [
            "Black rot (Botryosphaeria obtusa) infects fruit, leaves and bark; cankers on branches are a key inoculum source.",
            "Prune out cankers and dead wood, remove mummified fruit, and apply fungicides from bloom through summer.",
        ],
        "treatment": "Prune and destroy cankers and mummified fruit; apply captan or thiophanate-methyl fungicides during the season.",
        "economic": "Fruit rot and limb cankers reduce yield and can kill scaffold branches if cankers are left untreated.",
        "citations": [_C["apsnet"], _C["psu"]],
    },
    "Apple___Cedar_apple_rust": {
        "severity": "moderate",
        "pathogen": "Fungus (Gymnosporangium juniperi-virginianae)",
        "summary": "Cedar-apple rust needs both apple and juniper hosts and produces bright orange-yellow leaf and fruit lesions.",
        "passages": [
            "Cedar-apple rust alternates between juniper (cedar) and apple hosts; spores from cedar galls infect apple in spring.",
            "Plant resistant cultivars, remove nearby junipers where feasible, and apply fungicides at the pink-bud to petal-fall stages.",
        ],
        "treatment": "Use resistant cultivars; apply fungicides (e.g. myclobutanil) from pink bud through early summer; remove nearby galls on junipers.",
        "economic": "Heavy infection causes defoliation and blemished fruit, lowering marketable yield.",
        "citations": [_C["ipm"], _C["apsnet"]],
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "moderate",
        "pathogen": "Fungus (Podosphaera clandestina)",
        "summary": "Powdery mildew produces white fungal growth on leaves and fruit, distorting new growth.",
        "passages": [
            "Cherry powdery mildew thrives in warm days with high humidity but dry leaf surfaces, colonizing young leaves and fruit.",
            "Manage with resistant rootstocks/cultivars, canopy pruning for airflow, and sulfur or DMI fungicides.",
        ],
        "treatment": "Improve airflow by pruning; apply sulfur or DMI fungicides starting at shuck-fall and protect fruit through harvest.",
        "economic": "Affected fruit are downgraded; severe infection reduces packout and storability.",
        "citations": [_C["ipm"], _C["psu"]],
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "high",
        "pathogen": "Fungus (Cercospora zeae-maydis)",
        "summary": "Gray leaf spot forms rectangular gray-to-tan lesions that coalesce and destroy leaf area, cutting grain fill.",
        "passages": [
            "Gray leaf spot is favored by warm, humid conditions and continuous corn with conservation tillage that leaves infected residue.",
            "Resistant hybrids are the foundation of control; crop rotation, residue management, and timely fungicide at tasseling reduce losses.",
        ],
        "treatment": "Plant resistant hybrids, rotate crops and manage residue; apply foliar fungicide (e.g. strobilurin/triazole) around tasseling if disease pressure is high.",
        "economic": "Severe gray leaf spot can cut grain yield by 20-40% by reducing photosynthetic leaf area during grain fill.",
        "citations": [_C["apsnet"], _C["cornell"]],
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "moderate",
        "pathogen": "Fungus (Puccinia sorghi)",
        "summary": "Common rust forms cinnamon-brown pustules on both leaf surfaces; usually minor in field corn but damaging in sweet corn.",
        "passages": [
            "Common rust spores blow in from southern regions; cool, moist conditions favor development.",
            "Most field hybrids carry adequate resistance; susceptible sweet corn may need fungicide.",
        ],
        "treatment": "Grow resistant hybrids; in susceptible sweet corn apply foliar fungicide when pustules first appear.",
        "economic": "Generally low impact in field corn; can reduce sweet corn quality and yield when severe.",
        "citations": [_C["apsnet"], _C["ipm"]],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "high",
        "pathogen": "Fungus (Exserohilum turcicum)",
        "summary": "Northern leaf blight produces long cigar-shaped gray-green lesions that can blight whole leaves.",
        "passages": [
            "Northern leaf blight is favored by moderate temperatures and extended leaf wetness; it overwinters in residue.",
            "Resistant hybrids, rotation and residue management are primary controls, with fungicide if lesions appear before tasseling.",
        ],
        "treatment": "Use resistant hybrids, rotate away from corn, and apply fungicide if lesions reach the ear leaf before silking.",
        "economic": "Early, severe infection can reduce yield by up to 30% through loss of green leaf area.",
        "citations": [_C["apsnet"], _C["cornell"]],
    },
    "Grape___Black_rot": {
        "severity": "high",
        "pathogen": "Fungus (Guignardia bidwellii)",
        "summary": "Grape black rot rots berries into hard black mummies and spots leaves; one of the most destructive grape diseases in warm humid climates.",
        "passages": [
            "Black rot overwinters in mummified berries and cane lesions; spring rains spread spores to new growth.",
            "Sanitation (removing mummies), canopy management, and a protective fungicide program from early shoot growth through veraison are essential.",
        ],
        "treatment": "Remove mummified berries, apply fungicides (e.g. mancozeb, myclobutanil) from 1-3 inch shoots through fruit set, and open the canopy.",
        "economic": "Can destroy nearly the entire crop in wet seasons if left unmanaged.",
        "citations": [_C["cornell"], _C["apsnet"]],
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "high",
        "pathogen": "Fungal complex (esca / trunk disease)",
        "summary": "Esca (black measles) is a chronic trunk disease causing tiger-stripe leaves, berry spotting, and sudden vine collapse.",
        "passages": [
            "Esca is caused by a complex of wood-colonizing fungi entering through pruning wounds; symptoms worsen over years.",
            "There is no curative spray; manage by protecting pruning wounds, removing infected wood, and avoiding vine stress.",
        ],
        "treatment": "Protect large pruning wounds, prune during dry weather, remove and destroy dead arms/vines; no effective curative fungicide.",
        "economic": "Progressive vine decline reduces productive lifespan and forces costly replanting.",
        "citations": [_C["ipm"], _C["apsnet"]],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "moderate",
        "pathogen": "Fungus (Pseudocercospora vitis)",
        "summary": "Isariopsis leaf blight causes angular dark leaf spots that can drive late-season defoliation.",
        "passages": [
            "Leaf blight develops in warm, humid late-season conditions and is worse in dense, poorly ventilated canopies.",
            "Canopy management and the standard grape fungicide program generally keep it in check.",
        ],
        "treatment": "Improve canopy airflow and maintain the seasonal fungicide program; remove heavily infected leaves.",
        "economic": "Late defoliation reduces sugar accumulation and weakens vines for the next season.",
        "citations": [_C["cornell"], _C["cabi"]],
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "severe",
        "pathogen": "Bacterium (Candidatus Liberibacter) spread by Asian citrus psyllid",
        "summary": "Huanglongbing (citrus greening) is the most destructive citrus disease worldwide: blotchy mottled leaves, lopsided bitter fruit, and eventual tree death.",
        "passages": [
            "HLB is caused by a phloem-limited bacterium vectored by the Asian citrus psyllid; infected trees decline and there is no cure.",
            "Management focuses on psyllid control, removal of infected trees, and planting certified disease-free nursery stock.",
        ],
        "treatment": "No cure: control the Asian citrus psyllid vector, remove infected trees promptly, and use certified clean planting stock.",
        "economic": "HLB has devastated citrus industries (e.g. Florida lost the majority of production), making it the highest-impact disease in this set.",
        "citations": [_C["fao"], _C["apsnet"]],
    },
    "Peach___Bacterial_spot": {
        "severity": "moderate",
        "pathogen": "Bacterium (Xanthomonas arboricola pv. pruni)",
        "summary": "Bacterial spot causes angular leaf lesions, shot-holing, and sunken fruit spots in peaches.",
        "passages": [
            "Bacterial spot is worse on sandy, exposed sites and susceptible cultivars; it spreads in warm, wet, windy weather.",
            "Resistant cultivars plus copper and antibiotic sprays reduce but rarely eliminate the disease.",
        ],
        "treatment": "Plant resistant cultivars; apply copper-based sprays at low rates (to avoid phytotoxicity) plus oxytetracycline in high-pressure orchards.",
        "economic": "Fruit spotting reduces fresh-market grade; severe defoliation weakens trees.",
        "citations": [_C["psu"], _C["apsnet"]],
    },
    "Pepper,_bell___Bacterial_spot": {
        "severity": "high",
        "pathogen": "Bacterium (Xanthomonas spp.)",
        "summary": "Bacterial spot of pepper causes water-soaked leaf spots, defoliation, and raised scabby fruit lesions.",
        "passages": [
            "Bacterial spot is seedborne and splash-spread; warm, wet conditions and overhead irrigation accelerate epidemics.",
            "Use certified clean seed/transplants, resistant varieties, copper-based bactericides, and avoid working plants when wet.",
        ],
        "treatment": "Start with certified disease-free seed and resistant varieties; apply copper (often with mancozeb) preventively and avoid overhead irrigation.",
        "economic": "Defoliation exposes fruit to sunscald and scabby lesions make peppers unmarketable, with potential for major fresh-market losses.",
        "citations": [_C["ipm"], _C["cornell"]],
    },
    "Potato___Early_blight": {
        "severity": "moderate",
        "pathogen": "Fungus (Alternaria solani)",
        "summary": "Early blight forms concentric 'target-spot' lesions on older leaves, reducing canopy and tuber yield.",
        "passages": [
            "Early blight is favored by warm weather and alternating wet/dry periods and tends to hit stressed or aging foliage first.",
            "Balanced fertility, rotation, and protectant fungicides (chlorothalonil, mancozeb) manage the disease.",
        ],
        "treatment": "Maintain good fertility and irrigation, rotate crops, and apply protectant fungicides on a 7-10 day schedule under disease pressure.",
        "economic": "Progressive defoliation can reduce tuber size and yield by 20% or more in susceptible crops.",
        "citations": [_C["ipm"], _C["psu"]],
    },
    "Potato___Late_blight": {
        "severity": "severe",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "summary": "Late blight — the cause of the Irish potato famine — produces fast-spreading water-soaked lesions and can destroy a field in days under cool, wet conditions.",
        "passages": [
            "Phytophthora infestans spreads explosively in cool, wet, humid weather and can move from foliage to tubers, causing storage rot.",
            "Control requires resistant varieties, destruction of cull piles/volunteers, and timely protectant + systemic fungicides on a tight schedule.",
        ],
        "treatment": "Apply protectant fungicides preventively and switch to systemic products (e.g. with mandipropamid) once conditions favor disease; destroy infected plants and cull piles.",
        "economic": "Among the most economically damaging crop diseases in history; an unmanaged outbreak can cause total crop loss within a week.",
        "citations": [_C["apsnet"], _C["fao"]],
    },
    "Squash___Powdery_mildew": {
        "severity": "moderate",
        "pathogen": "Fungus (Podosphaera xanthii)",
        "summary": "Powdery mildew coats squash leaves in white fungal growth, causing premature senescence and sunscalded fruit.",
        "passages": [
            "Cucurbit powdery mildew spreads rapidly in warm conditions and does not require leaf wetness to infect.",
            "Resistant varieties, scouting leaf undersides, and rotating fungicide modes of action (sulfur, DMI, plus biologicals) manage resistance.",
        ],
        "treatment": "Grow resistant varieties, scout early, and rotate fungicide chemistries (sulfur, DMIs, biologicals) to avoid resistance.",
        "economic": "Early defoliation lowers yield and exposes fruit to sunscald, reducing marketable harvest.",
        "citations": [_C["ipm"], _C["cornell"]],
    },
    "Strawberry___Leaf_scorch": {
        "severity": "moderate",
        "pathogen": "Fungus (Diplocarpon earlianum)",
        "summary": "Leaf scorch produces dark purple blotches that merge and give leaves a scorched look, weakening plants.",
        "passages": [
            "Leaf scorch overwinters in infected leaves and spreads via splashing rain in warm, humid weather.",
            "Renovate beds, remove old infected foliage, improve airflow, and apply fungicides during bloom.",
        ],
        "treatment": "Renovate matted-row beds after harvest, remove infected leaves, improve airflow, and apply fungicides at bloom if needed.",
        "economic": "Reduced vigor lowers berry size and yield, especially in perennial matted-row systems.",
        "citations": [_C["psu"], _C["cabi"]],
    },
    "Tomato___Bacterial_spot": {
        "severity": "high",
        "pathogen": "Bacterium (Xanthomonas spp.)",
        "summary": "Bacterial spot causes small dark greasy leaf and fruit lesions, leading to defoliation and unmarketable fruit.",
        "passages": [
            "Bacterial spot is seedborne and splash-dispersed; warm, wet weather drives rapid spread.",
            "Use certified clean seed, resistant varieties, copper-based bactericides, and avoid overhead watering and handling wet plants.",
        ],
        "treatment": "Begin with clean seed and resistant varieties; apply copper plus mancozeb preventively and avoid overhead irrigation.",
        "economic": "Defoliation and fruit lesions can cause major fresh-market losses, particularly in humid production regions.",
        "citations": [_C["ipm"], _C["cornell"]],
    },
    "Tomato___Early_blight": {
        "severity": "moderate",
        "pathogen": "Fungus (Alternaria solani / A. linariae)",
        "summary": "Early blight forms concentric target-spot lesions on lower leaves and can cause stem and fruit lesions.",
        "passages": [
            "Early blight starts on older, lower leaves and progresses upward; warm wet conditions and plant stress accelerate it.",
            "Mulching, staking, rotation, and protectant fungicides limit spread.",
        ],
        "treatment": "Mulch and stake plants, rotate crops, remove lower infected leaves, and apply protectant fungicides (chlorothalonil/mancozeb) on a schedule.",
        "economic": "Defoliation reduces yield and exposes fruit to sunscald, cutting marketable production.",
        "citations": [_C["ipm"], _C["psu"]],
    },
    "Tomato___Late_blight": {
        "severity": "severe",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "summary": "Tomato late blight spreads explosively in cool wet weather, producing large water-soaked lesions and firm brown fruit rot that can destroy a crop within days.",
        "passages": [
            "Late blight (Phytophthora infestans) thrives in cool, moist, humid conditions and spreads on wind-blown spores over long distances.",
            "Remove volunteer plants and cull piles, grow resistant varieties, and apply protectant + systemic fungicides on a tight schedule once conditions favor disease.",
        ],
        "treatment": "Destroy infected plants promptly; apply protectant fungicides preventively and systemic products when weather favors disease; avoid overhead irrigation.",
        "economic": "One of the most destructive tomato diseases — an untreated outbreak can wipe out an entire field in under a week.",
        "citations": [_C["apsnet"], _C["cornell"]],
    },
    "Tomato___Leaf_Mold": {
        "severity": "moderate",
        "pathogen": "Fungus (Passalora fulva)",
        "summary": "Leaf mold causes pale yellow upper-leaf spots with olive-green velvety mold beneath, mainly in humid/greenhouse settings.",
        "passages": [
            "Leaf mold is driven by high humidity and poor airflow, especially in greenhouses and high tunnels.",
            "Lower humidity with ventilation, space plants, use resistant varieties, and apply fungicides where needed.",
        ],
        "treatment": "Reduce humidity through ventilation and spacing, grow resistant cultivars, and apply fungicides in protected culture.",
        "economic": "Severe leaf mold defoliates plants in greenhouses, reducing fruit set and yield.",
        "citations": [_C["cornell"], _C["cabi"]],
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "moderate",
        "pathogen": "Fungus (Septoria lycopersici)",
        "summary": "Septoria leaf spot forms many small circular spots with dark borders and gray centers, causing heavy lower-leaf defoliation.",
        "passages": [
            "Septoria overwinters in debris and spreads by splashing water; it favors warm, wet conditions.",
            "Sanitation, mulching, rotation, and protectant fungicides keep it from progressing up the canopy.",
        ],
        "treatment": "Remove infected lower leaves and debris, mulch, rotate crops, and apply protectant fungicides under wet conditions.",
        "economic": "Progressive defoliation reduces yield and fruit quality through sunscald exposure.",
        "citations": [_C["psu"], _C["ipm"]],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "moderate",
        "pathogen": "Arthropod pest (Tetranychus urticae)",
        "summary": "Two-spotted spider mites stipple leaves yellow/bronze and spin fine webbing; they explode in hot, dry, dusty conditions.",
        "passages": [
            "Spider mites thrive in hot, dry weather and are often flared by broad-spectrum insecticides that kill their natural predators.",
            "Conserve predatory mites, manage dust and water stress, and use selective miticides or insecticidal oils/soaps when thresholds are exceeded.",
        ],
        "treatment": "Preserve natural predators, reduce dust and drought stress, and apply selective miticides or horticultural oil/soap if populations exceed thresholds.",
        "economic": "Severe mite feeding reduces photosynthesis and can cause significant yield and quality loss in hot seasons.",
        "citations": [_C["ipm"], _C["cornell"]],
    },
    "Tomato___Target_Spot": {
        "severity": "moderate",
        "pathogen": "Fungus (Corynespora cassiicola)",
        "summary": "Target spot causes necrotic leaf lesions with concentric rings plus pitted fruit lesions, common in warm humid regions.",
        "passages": [
            "Target spot develops in warm, humid conditions with extended leaf wetness and dense canopies.",
            "Improve airflow, rotate crops, and apply fungicides; scout fruit as well as leaves.",
        ],
        "treatment": "Open the canopy, rotate crops, and apply protectant/systemic fungicides; monitor fruit for lesions.",
        "economic": "Both foliar and fruit lesions reduce yield and downgrade fresh-market fruit.",
        "citations": [_C["cabi"], _C["apsnet"]],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "severe",
        "pathogen": "Virus (TYLCV) spread by whitefly",
        "summary": "Tomato yellow leaf curl virus causes upward leaf curling, yellowing, stunting, and heavy flower drop, often devastating early-infected plants.",
        "passages": [
            "TYLCV is transmitted by the silverleaf/sweetpotato whitefly; early infection causes severe stunting and near-total yield loss.",
            "Use resistant varieties, whitefly control, reflective mulches, and remove infected plants; there is no cure once infected.",
        ],
        "treatment": "Plant TYLCV-resistant varieties, manage whitefly vectors (reflective mulch, insecticides, screening), and rogue infected plants. No curative treatment exists.",
        "economic": "Early infection can cause up to 100% yield loss, making it one of the most economically serious tomato diseases.",
        "citations": [_C["fao"], _C["apsnet"]],
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "high",
        "pathogen": "Virus (ToMV)",
        "summary": "Tomato mosaic virus causes mottled light/dark green leaves, leaf distortion, and internal fruit browning.",
        "passages": [
            "ToMV is extremely stable and spreads mechanically via hands, tools, and contaminated debris (not primarily by insects).",
            "Use resistant varieties and certified seed, sanitize hands/tools, and remove infected plants; milk or disinfectant rinses reduce mechanical spread.",
        ],
        "treatment": "Grow resistant varieties from certified seed, sanitize hands and tools, avoid tobacco use near plants, and remove infected plants. No curative treatment.",
        "economic": "Mottling, distortion, and internal fruit browning reduce both yield and marketable quality.",
        "citations": [_C["cornell"], _C["apsnet"]],
    },
}

# Generic 'healthy' grounding (any *_healthy class).
_HEALTHY = {
    "severity": "none",
    "pathogen": "None — foliage appears healthy",
    "summary": "The leaf appears healthy with no visible disease symptoms. Continue preventive monitoring and good cultural practices.",
    "passages": [
        "Healthy foliage shows uniform color without lesions, mottling, wilting, or pest stippling.",
        "Preventive practices — crop rotation, balanced nutrition, sanitation, and regular scouting — keep disease pressure low.",
    ],
    "treatment": "No treatment needed. Maintain scouting, balanced fertility/irrigation, sanitation, and rotation to keep plants healthy.",
    "economic": "A healthy crop protects full yield potential; preventive monitoring avoids costly outbreaks later in the season.",
    "citations": [_C["fao"], _C["ipm"]],
}


class FoundryIQ:
    """Grounds disease answers with cited knowledge (live Foundry IQ or local KB)."""

    def __init__(self):
        self.live = config.foundry_configured

    def ground(self, disease_label, query="severity, treatment and economic impact"):
        """Return GroundedKnowledge for a disease label.

        Tries live Foundry IQ retrieval when configured; always falls back to the
        curated, citation-backed local KB so the agent stays grounded offline.
        """
        if self.live:
            try:
                return self._ground_via_foundry(disease_label, query)
            except Exception as exc:  # network/credential failure -> graceful fallback
                print(f"[foundry] live retrieval failed, using local KB: {exc}")
        return self._ground_via_local_kb(disease_label)

    # --- Live Foundry IQ retrieval -----------------------------------------
    def _build_retrieve_request(self, disease_label, query):
        """Build the (readable, url, headers, payload) for the retrieve action.

        Shared by the live grounding path and the --selftest-live CLI so both
        send exactly the same request.
        """
        readable = disease_label.replace("___", " ").replace("_", " ")
        search_query = (
            f"{readable}: {query}. Provide pathogen, severity, recommended "
            f"treatment, and economic impact for this crop disease."
        )

        endpoint = config.foundry_endpoint.rstrip("/")
        url = (
            f"{endpoint}/knowledgebases/{config.foundry_kb_name}/retrieve"
            f"?api-version={config.foundry_api_version}"
        )
        headers = {"api-key": config.foundry_api_key,
                   "Content-Type": "application/json"}

        # Per-source params turn on the references array (needed for citations).
        ks_params = []
        if config.foundry_ks_name:
            ks_params.append({
                "knowledgeSourceName": config.foundry_ks_name,
                "kind": config.foundry_ks_kind,
                "includeReferences": True,
                "includeReferenceSourceData": True,
            })

        effort = (config.foundry_effort or "minimal").lower()
        if effort == "minimal":
            # No LLM query planning -> no Azure OpenAI charge (free-tier friendly).
            payload = {"intents": [{"type": "semantic", "search": search_query}]}
        else:
            # low/medium -> LLM-planned retrieval (uses the KB's default effort).
            payload = {"messages": [
                {"role": "assistant", "content": [{"type": "text", "text": (
                    "You retrieve agronomy knowledge about crop diseases. Return "
                    "content about pathogen, severity, treatment, and economic "
                    "impact.")}]},
                {"role": "user", "content": [{"type": "text", "text": search_query}]},
            ]}
        if ks_params:
            payload["knowledgeSourceParams"] = ks_params

        return readable, url, headers, payload

    def _ground_via_foundry(self, disease_label, query):
        """Query a Foundry IQ knowledge base via Azure AI Search agentic retrieval.

        Calls the knowledge base "retrieve" action:
            POST {search}/knowledgebases/{kb}/retrieve?api-version=...
        with header `api-key`. The engine plans the query (optional LLM), runs
        hybrid/semantic search across the knowledge base's sources, and returns
        grounded chunks plus a references array used for citations.

        Required env: FOUNDRY_IQ_ENDPOINT (Search URL), FOUNDRY_IQ_API_KEY,
        FOUNDRY_IQ_KNOWLEDGE_BASE. Optional: FOUNDRY_IQ_KNOWLEDGE_SOURCE/_KIND,
        FOUNDRY_IQ_REASONING_EFFORT, FOUNDRY_IQ_API_VERSION.
        """
        readable, url, headers, payload = self._build_retrieve_request(
            disease_label, query
        )
        resp = requests.post(url, json=payload, headers=headers,
                             timeout=config.request_timeout)
        resp.raise_for_status()
        data = resp.json()

        passages, citations = self._parse_foundry_response(data)
        if not passages:
            raise ValueError("Foundry IQ returned no grounded passages")

        return GroundedKnowledge(
            disease=readable,
            summary=passages[0],
            passages=passages,
            citations=citations,
            source="foundry-iq",
            grounded=True,
        )

    @staticmethod
    def _parse_foundry_response(data):
        """Extract passages + citations from an agentic-retrieval response.

        Grounded content arrives as a JSON-encoded string at
        response[0].content[0].text (a list of chunks with title/terms/content).
        The references array carries docKey/sourceData for citation linking.
        """
        passages, citations = [], []

        # 1) Grounding chunks (the unified grounding string).
        try:
            text_blob = data["response"][0]["content"][0]["text"]
            chunks = json.loads(text_blob)
        except (KeyError, IndexError, TypeError, ValueError):
            chunks = []

        titles_by_ref = {}
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                ref = str(chunk.get("ref_id", chunk.get("id", "")))
                title = _pretty_source_title(chunk.get("title"))
                titles_by_ref[ref] = title
                content = (chunk.get("content") or "").strip()
                if content:
                    passages.append(content)

        # 2) References -> citations (deduped by readable title).
        seen = set()
        for ref in data.get("references", []) or []:
            ref_id = str(ref.get("id", ""))
            doc_key = ref.get("docKey") or ref.get("dockey") or ""
            title = titles_by_ref.get(ref_id) or _pretty_source_title(doc_key)
            if title in seen:
                continue
            seen.add(title)
            citations.append(Citation(title, "Microsoft Foundry IQ", ""))

        # Fallback: no references array but we have chunks -> cite by chunk title.
        if not citations and titles_by_ref:
            for title in dict.fromkeys(titles_by_ref.values()):
                citations.append(Citation(title, "Microsoft Foundry IQ", ""))

        # Keep the demo/context tight: the top results are the most relevant.
        return passages[:8], citations[:6]

    # --- Local curated knowledge base --------------------------------------
    def _ground_via_local_kb(self, disease_label):
        readable = disease_label.replace("___", " — ").replace("_", " ")
        if disease_label.endswith("healthy"):
            entry = _HEALTHY
        else:
            entry = _KB.get(disease_label)

        if entry is None:
            # Unknown disease label: provide a safe generic grounded answer.
            return GroundedKnowledge(
                disease=readable,
                summary=f"{readable}: consult a local agricultural extension service for a confirmed diagnosis and region-specific treatment.",
                passages=[
                    "Confirm the diagnosis with a local extension specialist before applying treatments.",
                    "General good practice: remove symptomatic tissue, improve airflow, avoid overhead irrigation, and rotate crops.",
                ],
                citations=[_C["fao"], _C["ipm"]],
                source="local-kb",
                grounded=True,
            )

        return GroundedKnowledge(
            disease=readable,
            summary=entry["summary"],
            passages=entry["passages"],
            citations=entry["citations"],
            source="local-kb",
            grounded=True,
        )

    def get_entry(self, disease_label):
        """Return the raw KB entry dict (severity/pathogen/treatment/economic).

        Used by the reasoning agent's offline fallback. Returns the healthy or a
        generic entry when the label is unknown.
        """
        if disease_label.endswith("healthy"):
            return dict(_HEALTHY)
        return dict(_KB.get(disease_label, {
            "severity": "unknown",
            "pathogen": "Unknown pathogen",
            "summary": f"{disease_label.replace('___', ' — ').replace('_', ' ')} detected.",
            "treatment": "Consult a local agricultural extension service for a confirmed diagnosis and treatment plan.",
            "economic": "Impact varies; early diagnosis and intervention limit losses.",
            "citations": [_C["fao"], _C["ipm"]],
        }))

    # --- Live wiring diagnostics --------------------------------------------
    def selftest_live(self, disease_label="Tomato___Late_blight",
                      query="severity, treatment and economic impact"):
        """Ping the configured Foundry IQ knowledge base and dump the result.

        Validates the live Azure AI Search agentic-retrieval wiring end to end:
        prints the resolved config, the exact request, the raw HTTP response,
        and the parsed passages/citations. Returns True on success.
        """
        import time

        def mask(secret):
            if not secret:
                return "(unset)"
            return secret[:4] + "…" + secret[-4:] if len(secret) > 8 else "****"

        print("=" * 72)
        print("  Foundry IQ — LIVE wiring self-test")
        print("=" * 72)

        if not config.foundry_configured:
            print("✗ Not configured. Set these in your .env (see .env.example):")
            print("    FOUNDRY_IQ_ENDPOINT=https://<service>.search.windows.net")
            print("    FOUNDRY_IQ_API_KEY=<azure-ai-search-key>")
            print("    FOUNDRY_IQ_KNOWLEDGE_BASE=<knowledge-base-name>")
            print("\n  (Offline grounding still works without these.)")
            return False

        print(f"  Endpoint:        {config.foundry_endpoint}")
        print(f"  Knowledge base:  {config.foundry_kb_name}")
        print(f"  Knowledge src:   {config.foundry_ks_name or '(none — all KB sources)'}"
              f"  kind={config.foundry_ks_kind}")
        print(f"  Reasoning effort:{config.foundry_effort}")
        print(f"  API version:     {config.foundry_api_version}")
        print(f"  API key:         {mask(config.foundry_api_key)}")
        print(f"  Query disease:   {disease_label}")

        readable, url, headers, payload = self._build_retrieve_request(
            disease_label, query
        )
        print("-" * 72)
        print(f"  POST {url}")
        print("  Request body:")
        print("    " + json.dumps(payload)[:600])
        print("-" * 72)

        try:
            start = time.time()
            resp = requests.post(url, json=payload, headers=headers,
                                 timeout=config.request_timeout)
            elapsed = time.time() - start
        except requests.RequestException as exc:
            print(f"✗ Request failed (network/connection): {exc}")
            return False

        print(f"  HTTP {resp.status_code}  ({elapsed:.2f}s)")
        if resp.status_code != 200:
            print("✗ Non-200 response. Body:")
            print("    " + resp.text[:1000])
            self._explain_status(resp.status_code)
            return False

        try:
            data = resp.json()
        except ValueError:
            print("✗ Response was not JSON. Body:")
            print("    " + resp.text[:1000])
            return False

        print("  Raw response (truncated):")
        print("    " + json.dumps(data)[:800])
        print("-" * 72)

        passages, citations = self._parse_foundry_response(data)
        print(f"  Parsed: {len(passages)} passage(s), {len(citations)} citation(s)")
        for i, p in enumerate(passages[:5], 1):
            print(f"   [{i}] {p[:90]}")
        for c in citations[:5]:
            print(f"    • {c.title}  ({c.url or 'no docKey'})")

        if not passages:
            print("\n⚠ Connected, but no grounded passages came back. Likely causes:")
            print("    - The knowledge base has no indexed documents yet.")
            print("    - includeReferences/source name mismatch (set FOUNDRY_IQ_KNOWLEDGE_SOURCE).")
            print("    - Results fell below the reranker threshold.")
            return False

        print("\n✓ Live Foundry IQ retrieval is working.")
        return True

    @staticmethod
    def _explain_status(code):
        """Print a hint for common agentic-retrieval HTTP error codes."""
        hints = {
            401: "401 Unauthorized — check FOUNDRY_IQ_API_KEY (Azure AI Search key).",
            403: "403 Forbidden — the identity lacks the 'Search Index Data Reader' role.",
            404: "404 Not Found — check the knowledge base name and that the "
                 "endpoint is the Search URL (https://<svc>.search.windows.net).",
            400: "400 Bad Request — try FOUNDRY_IQ_API_VERSION=2026-04-01 for "
                 "'minimal' effort, or verify the knowledge source kind/name.",
            502: "502 Bad Gateway — a knowledge source failed (failOnError); "
                 "check the source connection in the Foundry portal.",
        }
        if code in hints:
            print(f"  Hint: {hints[code]}")


# Singleton
foundry_iq = FoundryIQ()


if __name__ == "__main__":
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Foundry IQ grounding — offline self-test or live wiring check"
    )
    parser.add_argument(
        "--selftest-live", action="store_true",
        help="Ping the configured Foundry IQ knowledge base and dump the raw "
             "response (validates live Azure AI Search agentic-retrieval wiring)."
    )
    parser.add_argument(
        "--label", default="Tomato___Late_blight",
        help="Disease class label to query (default: Tomato___Late_blight)."
    )
    parser.add_argument(
        "--query", default="severity, treatment and economic impact",
        help="Retrieval query text."
    )
    args = parser.parse_args()

    if args.selftest_live:
        ok = foundry_iq.selftest_live(disease_label=args.label, query=args.query)
        sys.exit(0 if ok else 1)

    # Default: offline grounding self-test (no network required).
    print("Testing Foundry IQ grounding (offline)...\n")
    print(f"Live Foundry IQ configured: {foundry_iq.live}")
    print("(run with --selftest-live to test a real knowledge base)\n")
    for label in [args.label, "Orange___Haunglongbing_(Citrus_greening)",
                  "Apple___healthy"]:
        g = foundry_iq.ground(label)
        print("=" * 70)
        print(f"{label}  [source: {g.source}]")
        print("-" * 70)
        print(g.as_context())
        print()
    print("✓ Foundry IQ module ready!")
