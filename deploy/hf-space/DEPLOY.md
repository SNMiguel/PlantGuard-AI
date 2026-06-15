# Deploy PlantGuard-AI to a Hugging Face Docker Space

This folder contains everything a Hugging Face **Docker** Space needs:
- `Dockerfile` — clones the public repo, installs CPU PyTorch + deps, pulls the
  model checkpoint from the v1.0.0 GitHub Release, and serves `api.py` on port 7860.
- `README.md` — the Space config (the YAML frontmatter is required; keep it verbatim).

## 1. Create the Space
1. Go to https://huggingface.co/new-space
2. Settings:
   - **Space name:** `plantguard-ai`
   - **License:** MIT
   - **SDK:** Docker → **Blank**
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public
3. Click **Create Space** (this makes an empty git repo with a starter Dockerfile + README).

## 2. Add these two files

### Option A — web UI (no git)
- On the Space's **Files** tab, upload `Dockerfile` (replace the starter one).
- Open `README.md` → **Edit** → replace all content with this folder's `README.md`.
- The Space rebuilds automatically on each change.

### Option B — git
```bash
git clone https://huggingface.co/spaces/<your-username>/plantguard-ai
cd plantguard-ai
# copy Dockerfile and README.md from this folder into the clone, overwriting
git add Dockerfile README.md
git commit -m "PlantGuard-AI Docker Space"
git push
```
Auth on push: username + a Hugging Face access token (Settings → Access Tokens) as the
password, or run `huggingface-cli login` first.

## 3. Watch the build
- The first build takes several minutes (installs PyTorch, downloads the 128 MB checkpoint).
- Follow the **Logs** tab. When it says the app is running on 7860, the demo is live at
  `https://huggingface.co/spaces/<your-username>/plantguard-ai`.

## Notes
- **No credentials are required.** With no secrets set, reasoning and grounding use the
  offline curated knowledge base, so the demo always works.
- To enable the live LLM / Foundry IQ on the Space, add the variables from `.env.example`
  as **Space secrets** (Settings → Variables and secrets). For a public demo, leaving it on
  the offline fallback avoids exposing your API quota.
- The Dockerfile pins the checkpoint to release `v1.0.0`. Bump that URL if you publish a new
  release.
