import io
import os

import requests
import torch
from PIL import ImageChops, ImageEnhance, Image
from dotenv import load_dotenv
from torchvision import models
from torchvision import transforms

load_dotenv()


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def compute_ela(img, quality=75, amplify=10):
    # re-save at lower quality and compute difference so tampered regions show brighter
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    ela = ImageChops.difference(img, recompressed)
    ela = ImageEnhance.Brightness(ela).enhance(amplify)
    return ela


def get_llm_summary(vendor, date, total, is_forged, ocr_text):
    verdict = "FORGED/SUSPICIOUS" if is_forged else "AUTHENTIC"
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv("CHAT_GROQ_API_KEY")}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "You are a document fraud analyst. Be brief and specific."},
                    {"role": "user", "content": (
                        f"This receipt is {verdict}.\n"
                        f"Vendor: {vendor or 'not found'}\n"
                        f"Date: {date or 'not found'}\n"
                        f"Total: {total or 'not found'}\n"
                        f"OCR text: {ocr_text[:300]}\n\n"
                        f"In 2-3 sentences explain why this receipt is {verdict}."
                    )}
                ],
                "max_tokens": 150,
            },
            timeout=30,
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"AI summary unavailable: {e}"
