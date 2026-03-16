import io

import torch
from PIL import ImageChops, ImageEnhance, Image
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from torchvision import models
from torchvision import transforms


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


def load_llm():
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
    return Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False, chat_format="chatml")


def get_llm_summary(llm, vendor, date, total, is_forged, ocr_text):
    try:
        verdict = "FORGED/SUSPICIOUS" if is_forged else "AUTHENTIC"

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a document fraud analyst. Give short, specific analysis only."
                },
                {
                    "role": "user",
                    "content": (
                        f"This receipt was classified as {verdict}.\n"
                        f"Vendor: {vendor or 'not found'}\n"
                        f"Date: {date or 'not found'}\n"
                        f"Total: {total or 'not found'}\n"
                        f"OCR text (first 300 chars):\n{ocr_text[:300]}\n\n"
                        f"In 2-3 sentences, explain specifically why this receipt is {verdict}."
                    )
                }
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Summary unavailable: {e}"
