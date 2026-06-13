"""
Central configuration for PlantGuard-AI's agent + Foundry IQ layers.

Reads settings from environment variables, with optional loading from a local
`.env` file (parsed manually so we don't add a python-dotenv dependency).

Nothing here is required to run the demo: when no LLM / Foundry IQ credentials
are configured, agent.py and foundry.py fall back to a curated offline
knowledge base so the app is always demoable.
"""
import os
from pathlib import Path

_ENV_LOADED = False


def load_env(path=".env"):
    """Load KEY=VALUE pairs from a .env file into os.environ (without override)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get(*names, default=None):
    """Return the first environment variable that is set among `names`."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


class Config:
    """Resolved runtime configuration (read once at import)."""

    def __init__(self):
        load_env()

        # --- LLM (Azure AI Foundry model deployment OR any OpenAI-compatible API) ---
        self.llm_api_key = _get(
            "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"
        )
        # Azure endpoint looks like https://<resource>.openai.azure.com
        # OpenAI-compatible base_url looks like https://api.openai.com/v1
        self.llm_endpoint = _get(
            "AZURE_OPENAI_ENDPOINT", "OPENAI_BASE_URL", "LLM_ENDPOINT"
        )
        self.llm_deployment = _get(
            "AZURE_OPENAI_DEPLOYMENT", "OPENAI_MODEL", "LLM_MODEL",
            default="gpt-4o-mini",
        )
        self.azure_api_version = _get(
            "AZURE_OPENAI_API_VERSION", default="2024-08-01-preview"
        )

        # --- Microsoft Foundry IQ (grounded knowledge retrieval) ---
        # Foundry IQ knowledge bases are backed by Azure AI Search agentic
        # retrieval. The "endpoint" is the Search service URL, the "key" is a
        # Search admin/query key, and we query a named knowledge base.
        self.foundry_endpoint = _get(
            "FOUNDRY_IQ_ENDPOINT", "AZURE_SEARCH_ENDPOINT", "FOUNDRY_PROJECT_ENDPOINT"
        )
        self.foundry_api_key = _get(
            "FOUNDRY_IQ_API_KEY", "AZURE_SEARCH_API_KEY", "FOUNDRY_API_KEY"
        )
        self.foundry_kb_name = _get(
            "FOUNDRY_IQ_KNOWLEDGE_BASE", "FOUNDRY_IQ_INDEX",
            default="plantguard-agri-kb",
        )
        # Optional: name + kind of a specific knowledge source (enables the
        # references array for citations). kind: searchIndex|azureBlob|web|sharepoint
        self.foundry_ks_name = _get("FOUNDRY_IQ_KNOWLEDGE_SOURCE")
        self.foundry_ks_kind = _get("FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND",
                                    default="azureBlob")
        # Retrieval reasoning effort: "minimal" (no LLM, free-tier friendly),
        # "low", or "medium" (LLM query planning -> Azure OpenAI charges).
        self.foundry_effort = _get("FOUNDRY_IQ_REASONING_EFFORT", default="minimal")
        self.foundry_api_version = _get("FOUNDRY_IQ_API_VERSION",
                                        default="2026-05-01-preview")

        self.request_timeout = int(_get("PLANTGUARD_TIMEOUT", default="30"))

    @property
    def is_azure(self):
        """True if the configured endpoint is an Azure OpenAI resource."""
        return bool(self.llm_endpoint and "openai.azure.com" in self.llm_endpoint)

    @property
    def llm_configured(self):
        return bool(self.llm_api_key and self.llm_endpoint)

    @property
    def foundry_configured(self):
        return bool(self.foundry_endpoint and self.foundry_api_key)


# Singleton used across modules.
config = Config()
