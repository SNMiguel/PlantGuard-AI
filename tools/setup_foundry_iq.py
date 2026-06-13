"""
Stage 2: provision the Foundry IQ knowledge base on Azure AI Search.

Builds the full agentic-retrieval pipeline from the blob container created in
Stage 1, using the Search REST API (no embedding model / no LLM deployment —
keyword + semantic search, "minimal" reasoning effort, so it stays free-tier):

    blob container
        -> data source      (azureblob)
        -> index            (content/title + semantic config)
        -> indexer          (pulls blobs into the index)
        -> knowledge source (searchIndex, points at the index)
        -> knowledge base   (minimal effort, extractive) <- this is Foundry IQ

Reads connection details from environment variables:
    AZURE_SEARCH_ENDPOINT   https://<service>.search.windows.net
    AZURE_SEARCH_API_KEY    Search admin key
    AZURE_STORAGE_ACCOUNT   storage account name
    AZURE_STORAGE_KEY       storage account key
    AZURE_STORAGE_CONTAINER blob container name (default: knowledge)

Run (PowerShell), reusing your Stage 1 variables:
    $env:AZURE_SEARCH_ENDPOINT="https://$SEARCH.search.windows.net"
    $env:AZURE_SEARCH_API_KEY=$SEARCHKEY
    $env:AZURE_STORAGE_ACCOUNT=$STORAGE
    $env:AZURE_STORAGE_KEY=$SAKEY
    $env:AZURE_STORAGE_CONTAINER=$CONTAINER
    venv\Scripts\python tools\setup_foundry_iq.py
"""
import os
import sys
import time

import requests

# --- Object names (match config.py / .env defaults) ---
DATASOURCE = "plantguard-blob-ds"
INDEX = "plantguard-index"
INDEXER = "plantguard-indexer"
SEMANTIC_CONFIG = "plantguard-semantic"
KNOWLEDGE_SOURCE = "plantguard-agri-source"
KNOWLEDGE_BASE = "plantguard-agri-kb"

API_STABLE = "2024-07-01"            # data source / index / indexer
API_PREVIEW = "2026-05-01-preview"   # knowledge source / knowledge base


def env(name, default=None, required=True):
    val = os.environ.get(name, default)
    if required and not val:
        sys.exit(f"ERROR: missing required environment variable {name}")
    return val


def put(endpoint, key, path, api_version, body, label):
    url = f"{endpoint.rstrip('/')}/{path}?api-version={api_version}"
    resp = requests.put(url, json=body,
                        headers={"api-key": key, "Content-Type": "application/json"},
                        timeout=60)
    if resp.status_code not in (200, 201, 204):
        print(f"  ✗ {label} failed: HTTP {resp.status_code}")
        print("    " + resp.text[:1200])
        resp.raise_for_status()
    print(f"  ✓ {label}")


def main():
    endpoint = env("AZURE_SEARCH_ENDPOINT")
    key = env("AZURE_SEARCH_API_KEY")
    account = env("AZURE_STORAGE_ACCOUNT")
    storage_key = env("AZURE_STORAGE_KEY")
    container = env("AZURE_STORAGE_CONTAINER", default="knowledge", required=False)

    conn = (f"DefaultEndpointsProtocol=https;AccountName={account};"
            f"AccountKey={storage_key};EndpointSuffix=core.windows.net")

    print("Provisioning Foundry IQ knowledge base on Azure AI Search...")
    print(f"  Search: {endpoint}")
    print(f"  Blob:   {account}/{container}\n")

    # 1) Data source (blob)
    put(endpoint, key, f"datasources/{DATASOURCE}", API_STABLE, {
        "name": DATASOURCE,
        "type": "azureblob",
        "credentials": {"connectionString": conn},
        "container": {"name": container},
    }, "Data source")

    # 2) Index (text + semantic config, no vectors)
    put(endpoint, key, f"indexes/{INDEX}", API_STABLE, {
        "name": INDEX,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True,
             "searchable": False, "filterable": True, "retrievable": True},
            {"name": "content", "type": "Edm.String",
             "searchable": True, "retrievable": True},
            {"name": "title", "type": "Edm.String",
             "searchable": True, "retrievable": True},
            {"name": "filepath", "type": "Edm.String",
             "searchable": False, "retrievable": True},
        ],
        "semantic": {
            "configurations": [{
                "name": SEMANTIC_CONFIG,
                "prioritizedFields": {
                    "titleField": {"fieldName": "title"},
                    "prioritizedContentFields": [{"fieldName": "content"}],
                    "prioritizedKeywordsFields": [],
                },
            }],
        },
    }, "Index")

    # 3) Indexer (blob -> index)
    put(endpoint, key, f"indexers/{INDEXER}", API_STABLE, {
        "name": INDEXER,
        "dataSourceName": DATASOURCE,
        "targetIndexName": INDEX,
        "parameters": {"configuration": {
            "parsingMode": "default",
            "dataToExtract": "contentAndMetadata",
        }},
        "fieldMappings": [
            {"sourceFieldName": "metadata_storage_path", "targetFieldName": "id",
             "mappingFunction": {"name": "base64Encode"}},
            {"sourceFieldName": "metadata_storage_name", "targetFieldName": "title"},
            {"sourceFieldName": "metadata_storage_path", "targetFieldName": "filepath"},
        ],
    }, "Indexer")

    # Run the indexer and wait for it to finish.
    run_url = f"{endpoint.rstrip('/')}/indexers/{INDEXER}/run?api-version={API_STABLE}"
    requests.post(run_url, headers={"api-key": key}, timeout=30)
    print("  … running indexer", end="", flush=True)
    status_url = f"{endpoint.rstrip('/')}/indexers/{INDEXER}/status?api-version={API_STABLE}"
    indexed = 0
    for _ in range(30):  # up to ~90s
        time.sleep(3)
        print(".", end="", flush=True)
        s = requests.get(status_url, headers={"api-key": key}, timeout=30).json()
        last = s.get("lastResult") or {}
        state = last.get("status")
        if state == "success":
            indexed = last.get("itemsProcessed", 0)
            break
        if state in ("transientFailure", "error"):
            print(f"\n  ✗ Indexer error: {last.get('errorMessage', last)}")
            break
    print(f"\n  ✓ Indexer finished — {indexed} documents indexed")

    # 4) Knowledge source (searchIndex)
    put(endpoint, key, f"knowledgesources/{KNOWLEDGE_SOURCE}", API_PREVIEW, {
        "name": KNOWLEDGE_SOURCE,
        "kind": "searchIndex",
        "description": "PlantGuard agronomy knowledge — 38 crop-disease docs with citations.",
        "searchIndexParameters": {
            "searchIndexName": INDEX,
            "semanticConfigurationName": SEMANTIC_CONFIG,
            "sourceDataFields": ["title", "filepath", "content"],
            "searchFields": ["content", "title"],
        },
    }, "Knowledge source")

    # 5) Knowledge base (minimal effort, extractive — no LLM, free-tier friendly).
    # outputMode is omitted: with "minimal" effort the response is extractive by
    # default, which is exactly what foundry.py parses.
    put(endpoint, key, f"knowledgebases/{KNOWLEDGE_BASE}", API_PREVIEW, {
        "name": KNOWLEDGE_BASE,
        "description": "Grounds crop-disease answers in PlantGuard's cited agronomy knowledge.",
        "knowledgeSources": [{"name": KNOWLEDGE_SOURCE}],
        "retrievalReasoningEffort": {"kind": "minimal"},
    }, "Knowledge base")

    print("\n" + "=" * 60)
    print("  ✓ Foundry IQ knowledge base is live!")
    print("=" * 60)
    print("Add these lines to your .env:\n")
    print(f"FOUNDRY_IQ_ENDPOINT={endpoint}")
    print("FOUNDRY_IQ_API_KEY=<your Search admin key>")
    print(f"FOUNDRY_IQ_KNOWLEDGE_BASE={KNOWLEDGE_BASE}")
    print(f"FOUNDRY_IQ_KNOWLEDGE_SOURCE={KNOWLEDGE_SOURCE}")
    print("FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND=searchIndex")
    print("FOUNDRY_IQ_REASONING_EFFORT=minimal")
    print("\nThen validate:  venv\\Scripts\\python foundry.py --selftest-live")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
