# Forgesight
Receipt forgery detection using ELA + MobileNetV2, with OCR extraction and a Streamlit dashboard.

**Live Demo:** https://forgesight.streamlit.app/

---

## What it does

Forgesight is an end-to-end receipt analysis pipeline built for the 2026 Rihal CodeStacker ML Challenge (DocFusion). It:

1. Extracts structured fields (vendor, date, total) from scanned receipts using OCR and regex
2. Detects forged or tampered receipts using Error Level Analysis (ELA) combined with MobileNetV2 visual features
3. Explains anomalies using Llama 3.1 8B via the Groq API
4. Visualises findings in an interactive Streamlit dashboard

---

## How it works

```
Receipt Image
      |
      +--> OCR (PaddleOCR / EasyOCR) --> extract vendor / date / total
      |
      +--> MobileNetV2 features (1280-dim)        ]
      |                                            +--> concat (2560-dim) --> LogisticRegression --> is_forged
      +--> ELA --> MobileNetV2 features (1280-dim) ]
                                                   |
                                                   +--> Groq API (Llama 3.1 8B) --> AI anomaly summary
```

ELA (Error Level Analysis) re-saves the image at lower JPEG quality and computes the pixel difference. Tampered regions show brighter because copy-paste edits introduce compression inconsistencies. MobileNetV2 features are extracted from both the original and ELA image, concatenated, and fed into a LogisticRegression classifier trained on the Find-It-Again dataset.

---

## Project Structure

```
forgesight/
|
├── notebooks/
|   ├── Level_1_EDA.ipynb
|   ├── Level_2_Extraction.ipynb
|   └── Level_3_Model_Training_Data_prep.ipynb
|
├── scripts/
|   └── fia_to_jsonl.py         # converts Find-It-Again csv to harness JSONL format
|
├── Dockerfile
├── packages.txt                # system dependencies for Streamlit Cloud
├── requirements.txt
├── requirements.lock
|
├── extractors.py               # vendor / date / total extraction functions
├── regex_helper.py             # all regex patterns
├── app.py                      # Streamlit UI
├── detection_helpers.py        # build_model, get_transform, compute_ela
├── solution.py                 # DocFusionSolution harness interface (Level 4)
|
├── forgery_model.pkl           # trained model
├── classification_report.py    # evaluate predictions against labels.jsonl
└── check_submission.py         # harness interface smoke test
```

---

## Setup

```bash
git clone https://github.com/yourusername/forgesight.git
cd forgesight
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Upload any receipt image to get extracted fields, forgery verdict, bounding boxes on known forged regions, and an AI-generated anomaly summary.

---

## Docker
```bash
docker build -t forgesight .
docker run -p 8501:8501 forgesight
```

Open `http://localhost:8501`.

> The Docker image includes a demo Groq API key for the LLM summary feature. No additional configuration needed. Just to make it easier for the judges, and it will be removed after evaluation. 

---

## Harness Validation

Generate the data first from the Find-It-Again dataset:

```bash
python scripts/fia_to_jsonl.py
```

Then run the checker:

```bash
python check_submission.py --submission . --data data
```

To evaluate prediction accuracy against ground truth labels:

```bash
python classification_report.py
```

---

## AI Anomaly Summary

Receipts are analyzed by **Llama 3.1 8B** via the [Groq API](https://console.groq.com), free tier, no credit card required, resets daily.

Each upload generates a 2-3 sentence human-readable explanation of why the receipt was flagged or cleared.

**Why not a local LLM?**
The original approach used **Qwen 2.5 0.5B GGUF** running locally via `llama-cpp-python`. It was extremely efficient, ~150ms inference on CPU, zero network dependency, fully offline. However `llama-cpp-python` requires compilation from source (`gcc`, `cmake`) which fails on Streamlit Cloud and Python 3.13 Docker environments where no C compiler is available. Pre-built wheels only exist for Python 3.12 and below. Switching to the Groq API solved the deployment problem with no loss in summary quality, Llama 3.1 8B is significantly more capable than Qwen 0.5B anyway.

**Rate limits (free tier):**
- ~30,000 tokens/minute for Llama 3.1 8B
- Resets every 24 hours
- No credit card required, on limit hit the app shows a graceful fallback message


## OCR Engine

The app uses **PaddleOCR** locally and in Docker, it produces better text extraction on receipts, especially for dense layouts and non-standard fonts.

On **Streamlit Cloud**, PaddleOCR crashes on the second prediction due to a known threading bug in the Paddle C++ runtime (tracked in [PaddlePaddle/PaddleOCR#16238](https://github.com/PaddlePaddle/PaddleOCR/issues/16238)). As a workaround, the cloud deployment falls back to **EasyOCR**, which is more stable but produces slightly noisier output, particularly on right-aligned values and small fonts. This means field extraction (vendor, date, total) may be less accurate on Streamlit Cloud compared to running locally.

To force EasyOCR locally (e.g. for testing), set the environment variable:
```bash
# Linux/macOS
USE_EASYOCR=true streamlit run app.py
# or for windows
set USE_EASYOCR=true && streamlit run app.py
```

---

## Performance

| Metric | Value |
|---|---|
| Training time (CPU) | ~4 minutes |
| Inference per image | ~1-2 seconds |
| Model size | 37 KB |
| Test accuracy | 74% |
| Forged F1 | 0.36 |
| Macro F1 | 0.61 |

---

## Known Limitations

**Model**
- Trained on ~577 Find-It-Again receipts (94 forged, 483 authentic). Small dataset limits generalization to unseen forgery styles and receipt layouts.
- Most effective on copy-paste and pixel-level tampering detectable via ELA. Forgeries that preserve JPEG compression history may not be detected.
- Two MobileNetV2 forward passes per image makes training ~4 minutes on CPU.

**Bounding Boxes**
- Boxes are shown only for images whose filename matches a training set image (ground-truth lookup). No boxes are shown for unseen receipts.

**OCR**
- Receipts with wide column layouts often have right-aligned numeric values dropped by the OCR engine, causing total to be null even when visually present.
- Low-resolution or skewed receipts produce noisy OCR output that may cause field extraction to fail.
- EasyOCR (used on Streamlit Cloud) is less accurate than PaddleOCR on receipt-specific layouts.

**OCR Field Extraction (Regex)**
- Regex patterns were designed and tested on the Find-It-Again (SROIE-based) dataset which consists primarily of Malaysian receipts. They may fail on receipts from different regions, currencies, or formats not seen during development.
- Total extraction relies on keyword matching ("TOTAL", "NET AMT", "RM" etc). Receipts that use non-standard labels or different languages for the total field will likely return null.
- Vendor extraction uses the first non-numeric OCR line as a heuristic, this fails when receipts have headers, logos, or document numbers printed above the vendor name.
- Date extraction covers many common formats (DD/MM/YYYY, YYYY-MM-DD, "OCT 3 2016" etc) but will miss unusual or locale-specific date formats.
- Improving extraction accuracy requires monitoring failures on new receipt types, identifying missing patterns, and iteratively expanding regex coverage.

**LLM Summary**
- Depends on the Groq API. summary is unavailable if the API is unreachable or rate limited.
- The LLM only sees extracted fields and the first 300 characters of OCR text. It has no visual understanding of the receipt.

---

## Tech Stack

| Component | Technology |
|---|---|
| OCR (local/Docker) | PaddleOCR |
| OCR (Streamlit Cloud) | EasyOCR |
| Visual features | MobileNetV2 (torchvision) |
| Forgery signal | Error Level Analysis (PIL) |
| Classifier | LogisticRegression (scikit-learn) |
| LLM summary | Llama 3.1 8B via Groq API |
| UI | Streamlit |
| Containerization | Docker |
