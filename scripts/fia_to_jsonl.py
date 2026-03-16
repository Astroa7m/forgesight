import ast
import csv
import json
import shutil
from pathlib import Path

import kagglehub

from extractors import extract_vendor, extract_total, extract_date

fia_root = kagglehub.dataset_download('nikita2998/find-it-again-dataset')

FINDIT_DIR = Path(f"{fia_root}/findit2")
OUT_DIR = Path("../data")


# def parse_bboxes(ann_str):
#     boxes = []
#     ann = ann_str.strip()
#     if ann in ("0", "nan", ""):
#         return boxes
#     try:
#         d = ast.literal_eval(ann)
#         for reg in d.get("regions", []):
#             sa = reg.get("shape_attributes", {})
#             ra = reg.get("region_attributes", {})
#             if str(ra.get("Original area", "no")).lower() == "yes":
#                 continue
#             if sa.get("name") == "rect":
#                 boxes.append([sa["x"], sa["y"], sa["width"], sa["height"]])
#     except Exception:
#         pass
#     return boxes


def parse_fraud_type(ann_str):
    ann = ann_str.strip()
    if ann in ("0", "nan", ""):
        return None
    try:
        d = ast.literal_eval(ann)
        for reg in d.get("regions", []):
            ra = reg.get("region_attributes", {})
            et = ra.get("Entity type", {})
            return et.lower() if isinstance(et, str) else None
    except Exception:
        pass
    return None


def convert_split(txt_file, split_name, include_labels=True):
    src_img_dir = FINDIT_DIR / split_name
    out_split = OUT_DIR / split_name
    out_img_dir = out_split / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    labels_for_test = []
    labels_path = out_split / "labels.jsonl"  # will only be used for test
    out_jsonl = out_split / f"{split_name}.jsonl"
    records = []

    with open(FINDIT_DIR / txt_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["image"].strip()
            src_path = src_img_dir / fname
            if not src_path.exists():
                continue

            dst_path = out_img_dir / fname
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

            record_id = fname.replace(".png", "").replace(".jpg", "")
            is_forged = int(row["forged"])

            ocr_path = src_img_dir / fname.replace(".png", ".txt").replace(".jpg", ".txt")
            ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
            ocr_lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]

            vendor = extract_vendor(ocr_lines)
            date = extract_date(ocr_text)
            total = extract_total(ocr_text)
            fraud_type = parse_fraud_type(row["forgery annotations"])
            rec = {
                "id": record_id,
                "image_path": f"images/{fname}",
                "fields": {
                    "vendor": vendor,
                    "date": date,
                    "total": total,
                },
            }

            if include_labels:
                rec["label"] = {
                    "is_forged": is_forged,
                    "fraud_type": fraud_type,
                }
            else:
                labels_for_test.append(
                    {
                        "id": record_id,
                        "label": {
                            "is_forged": is_forged,
                            "fraud_type": fraud_type,
                        }
                    }
                )

            records.append(rec)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    if not include_labels:
        with open(labels_path, "w", encoding="utf-8") as f:
            for r in labels_for_test:
                f.write(json.dumps(r) + "\n")

    print(f"  {split_name}: {len(records)} records → {out_jsonl}")
    return records


if __name__ == "__main__":
    print("converting fia dataset to JSONL format...")
    convert_split("train.txt", "train", include_labels=True)
    convert_split("test.txt",  "test",  include_labels=False)
    print(f"\nsaved under: {OUT_DIR}/")
