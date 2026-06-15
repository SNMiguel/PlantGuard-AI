# PlantGuard-AI — Hugging Face Docker Space
# Serves the FastAPI web app (api.py) + website on port 7860.
# It clones the public GitHub repo and pulls the model checkpoint from the
# v1.0.0 GitHub Release at build time, so this Space only needs this Dockerfile
# and the config README.md.

FROM python:3.11-slim

# System deps: git (to clone the repo), curl + CA certs (to fetch the checkpoint).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces expect a non-root user (uid 1000). Create and own the work
# dir as root first, so the unprivileged user can write into it (git clone, pip).
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown -R user:user /home/user/app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Clone the public project (shallow).
RUN git clone --depth 1 https://github.com/SNMiguel/PlantGuard-AI.git .

# Python deps: install CPU-only PyTorch first (avoids the multi-GB CUDA wheels),
# then the rest of requirements.txt (torch==2.1.0 is already satisfied).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Fetch the trained checkpoint from the GitHub Release (public asset).
RUN mkdir -p models/saved \
    && curl -fL -o models/saved/resnet18_best.pth \
       https://github.com/SNMiguel/PlantGuard-AI/releases/download/v1.0.0/resnet18_best.pth

# Spaces route traffic to port 7860.
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
