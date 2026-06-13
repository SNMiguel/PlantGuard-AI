"""
Inference script for PlantGuard-AI.

Loads the trained ResNet checkpoint and classifies a single leaf image into one
of the 38 PlantVillage disease classes, returning the predicted label, a
human-readable species/condition split, and calibrated confidence scores.

Usage (CLI):
    python predict.py path/to/leaf.jpg
    python predict.py path/to/leaf.jpg --top_k 5 --model resnet18

Usage (import):
    from predict import PlantDoctor
    doctor = PlantDoctor()              # loads models/saved/resnet18_best.pth
    result = doctor.predict("leaf.jpg")
    print(result["label"], result["confidence"])
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

# The existing modules print Unicode glyphs (e.g. "✓"). On Windows the default
# console codec (cp1252) can't encode them, so force UTF-8 output where possible.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from models.resnet_model import ResNetClassifier
from utils.transforms import PlantDiseaseTransforms

# Canonical class ordering. This MUST match torchvision.datasets.ImageFolder,
# which sorts class folder names alphabetically (see utils/data_loader.py).
# Hardcoded so inference works without the dataset present on disk.
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

DEFAULT_CHECKPOINT = Path("models/saved/resnet18_best.pth")


def split_label(raw_label):
    """Split a raw 'Species___Condition' label into readable parts.

    Returns:
        dict with keys: species, condition, is_healthy
    """
    species, _, condition = raw_label.partition("___")
    species = species.replace("_", " ").strip()
    condition_clean = condition.replace("_", " ").strip()
    is_healthy = condition.lower() == "healthy"
    return {
        "species": species,
        "condition": "Healthy" if is_healthy else condition_clean,
        "is_healthy": is_healthy,
    }


class PlantDoctor:
    """Wraps the trained ResNet classifier for single-image inference."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, model_name="resnet18",
                 class_names=None, img_size=224, device=None):
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # pretrained=False: we immediately overwrite weights with our checkpoint,
        # so there is no need to download ImageNet weights (keeps inference offline).
        self.classifier = ResNetClassifier(
            num_classes=self.num_classes,
            model_name=model_name,
            pretrained=False,
        )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Train a model first (python main.py) or point --checkpoint at a .pth file."
            )
        self.classifier.load_checkpoint(checkpoint_path)
        self.classifier.model.eval()

        self.transform = PlantDiseaseTransforms(img_size=img_size).get_val_transforms()

    def _to_tensor(self, image):
        """Accept a path, a PIL Image, or a numpy array; return a 1x3xHxW tensor."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:  # assume array-like (e.g. numpy from Gradio)
            image = Image.fromarray(image).convert("RGB")
        return self.transform(image).unsqueeze(0)

    def predict(self, image, top_k=3):
        """Classify an image.

        Args:
            image: file path, PIL.Image, or numpy array.
            top_k: number of ranked predictions to return.

        Returns:
            dict: {
                label, species, condition, is_healthy, confidence,
                class_index, top_k: [{label, species, condition, confidence}, ...]
            }
        """
        tensor = self._to_tensor(image)
        predictions, probabilities = self.classifier.predict(tensor)

        probs = probabilities.squeeze(0).cpu()
        class_index = int(predictions.item())
        raw_label = self.class_names[class_index]

        k = min(top_k, self.num_classes)
        top_probs, top_idx = torch.topk(probs, k)
        top_k_list = []
        for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
            label = self.class_names[idx]
            parts = split_label(label)
            top_k_list.append({
                "label": label,
                "species": parts["species"],
                "condition": parts["condition"],
                "confidence": round(float(p), 4),
            })

        parts = split_label(raw_label)
        return {
            "label": raw_label,
            "species": parts["species"],
            "condition": parts["condition"],
            "is_healthy": parts["is_healthy"],
            "confidence": round(float(probs[class_index]), 4),
            "class_index": class_index,
            "top_k": top_k_list,
        }


# Module-level singleton so callers (e.g. app.py) don't reload the model per call.
_DOCTOR = None


def get_doctor(checkpoint_path=DEFAULT_CHECKPOINT, model_name="resnet18"):
    """Return a cached PlantDoctor instance."""
    global _DOCTOR
    if _DOCTOR is None:
        _DOCTOR = PlantDoctor(checkpoint_path=checkpoint_path, model_name=model_name)
    return _DOCTOR


def predict_image(image, top_k=3):
    """Convenience one-liner using the cached doctor."""
    return get_doctor().predict(image, top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlantGuard-AI single-image inference")
    parser.add_argument("image", type=str, help="Path to a leaf image")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="Model architecture matching the checkpoint")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of ranked predictions to show")
    args = parser.parse_args()

    doctor = PlantDoctor(checkpoint_path=args.checkpoint, model_name=args.model)
    result = doctor.predict(args.image, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("  PlantGuard-AI Diagnosis")
    print("=" * 60)
    status = "HEALTHY" if result["is_healthy"] else "DISEASE DETECTED"
    print(f"  Plant:      {result['species']}")
    print(f"  Status:     {status}")
    print(f"  Condition:  {result['condition']}")
    print(f"  Confidence: {result['confidence'] * 100:.2f}%")
    print("-" * 60)
    print("  Top predictions:")
    for i, p in enumerate(result["top_k"], 1):
        print(f"   {i}. {p['species']} — {p['condition']:<28} {p['confidence'] * 100:6.2f}%")
    print("=" * 60 + "\n")
