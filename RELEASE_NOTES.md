# PlantGuard-AI v1.0.0: trained model checkpoint

This release ships the trained **ResNet18** weights (`resnet18_best.pth`) for PlantGuard-AI. The file is ~128 MB, which is too large for normal Git, so it is distributed here as a release asset instead of being committed to the repo.

## What's in the box
- `resnet18_best.pth` — the vision model that powers `predict.py`, `api.py`, and `app.py`.

## How to use it
1. Download `resnet18_best.pth` from the Assets below.
2. Place it at `models/saved/resnet18_best.pth` in your clone of the repo.
3. Install deps and run the app:
   ```bash
   pip install -r requirements.txt
   python api.py        # then open http://127.0.0.1:8000
   ```
   (No checkpoint, no dataset, and no cloud credentials are required to run inference; the offline fallbacks cover reasoning and grounding.)

## Model details
- Architecture: ResNet18 (transfer learning), classifier head sized for 38 classes.
- Coverage: 38 disease classes across 14 crop species (PlantVillage).
- Test accuracy: 95.02% · precision 95.75% · F1 95.01% on 8,147 held-out images.
- File size: 128.3 MB.
- SHA256: `634DAD9854776620784362B8F39862B4DA4032EC13A0A80AEF7FB6ADA81FD3C7`

## Verify the download (optional)
PowerShell:
```powershell
(Get-FileHash .\models\saved\resnet18_best.pth -Algorithm SHA256).Hash
```
The output should match the SHA256 above.
