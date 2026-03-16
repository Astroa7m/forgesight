import pickle

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR

from detection_helpers import build_model, get_transform, get_llm_summary, compute_ela
from extractors import extract_vendor, extract_date, extract_total

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_FILE = "forgery_model.pkl"  # could come from solution:train but what if the `work_dir` changed? let's have our own trained model form the train notebook


@st.cache_resource
def load_ocr_model():
    return PaddleOCR(use_angle_cls=True, lang='en')


@st.cache_resource
def load_everything():
    # loading mobilenet v2 model with default weights and resetting classifier head
    model = build_model()

    # loading model
    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)

    ocr_engine = load_ocr_model()

    return model, data["clf"], data.get("bbox_lookup", {}), ocr_engine


st.set_page_config(page_title="Receipt Forgery Detector", page_icon="🔍")
st.title("Receipt Forgery Detector")

model, clf, bbox_lookup, ocr_engine = load_everything()

# uploading
uploaded = st.file_uploader("Upload a receipt image", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.stop()

# converting the image to the same format we used during training
pil = Image.open(uploaded).convert("RGB")

with st.spinner("Running OCR..."):
    img_array = np.array(pil)

    result = ocr_engine.predict(img_array)

    lines = []
    for res in result:
        for box_info in res["rec_texts"]:
            lines.append(box_info)

    ocr_text = "\n".join(lines)
    # print(ocr_text)
vendor = extract_vendor(lines)
date = extract_date(ocr_text)
total = extract_total(ocr_text)

tf = get_transform()
with torch.no_grad():
    original_feat = model(tf(pil).unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()
    ela_feat = model(tf(compute_ela(pil)).unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()
feat = np.concatenate([original_feat, ela_feat])
pred = clf.predict([feat])[0]
proba = clf.predict_proba([feat])[0]  # [auth precent, forged percent]
bboxes = bbox_lookup.get(uploaded.name, [])
missing_total = total is None
# print(f"proba is, {proba}")
final_forged = int(pred) == 1

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Receipt")
    annotated = pil.copy()
    # drawing boxes
    if bboxes:
        draw = ImageDraw.Draw(annotated)
        for x, y, w, h in bboxes:
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    st.image(annotated, width="stretch")

with col2:
    st.subheader("Extracted Fields")
    st.markdown(f"**Vendor:** {vendor or '⚠️ not found'}")
    st.markdown(f"**Date:** {date or '⚠️ not found'}")
    st.markdown(f"**Total:** {total or '⚠️ not found'}")

    st.divider()
    st.subheader("Anomaly Status")

    if final_forged:
        st.error("⚠️ FORGED / SUSPICIOUS")
    else:
        st.success("✅ AUTHENTIC")

    st.progress(float(proba[1]), text=f"Forgery possibility: {proba[1]:.1%}")

    if bboxes:
        st.markdown("**Flagged regions (x, y, w, h):**")
        for b in bboxes:
            st.code(str(b))

st.divider()
st.subheader("🧠 AI Analysis")
with st.spinner("Generating analysis..."):
    summary = get_llm_summary(vendor, date, total, final_forged, ocr_text)
st.info(summary)

# output = {
#     "id": uploaded.name,
#     "vendor": vendor,
#     "date": date,
#     "total": total,
#     "is_forged": int(final_forged),
# }
