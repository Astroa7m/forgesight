import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from detection_helpers import build_model, get_transform, compute_ela


def _load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_features(backbone, img_paths):
    tf = get_transform()
    feats = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            with torch.no_grad():
                # original image features
                orig_feat = backbone(tf(img).unsqueeze(0)).squeeze().cpu().detach().numpy()

                # ELA image features highlightingtampered regions
                ela_feat = backbone(tf(compute_ela(img)).unsqueeze(0)).squeeze().cpu().detach().numpy()

            # concatenating boht
            feats.append(np.concatenate([orig_feat, ela_feat]))
        except Exception:
            feats.append(np.zeros(2560, dtype=np.float32))
    return np.array(feats)


class DocFusionSolution:

    def train(self, train_dir: str, work_dir: str) -> str:
        """
        Train a model on data in train_dir.

        Args:
            train_dir: Path to directory containing train.jsonl and images/
            work_dir:  Scratch directory for writing model artifacts

        Returns:
            Path to the saved model directory (typically inside work_dir)
        """

        train_dir = Path(train_dir)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        records = _load_jsonl(train_dir / "train.jsonl")
        print(f"[train] {len(records)} records")

        print(f"[train] images found: {sum(1 for r in records if (train_dir / r['image_path']).exists())}")
        print(f"[train] images missing: {sum(1 for r in records if not (train_dir / r['image_path']).exists())}")
        img_paths = [train_dir / r["image_path"] for r in records]
        labels = [int(r["label"]["is_forged"]) for r in records]

        genuine_totals = []
        bbox_lookup = {} # I love to see it work with the data that we trained on xD
        for r in records:
            if int(r["label"]["is_forged"]) == 0:
                total = r.get("fields", {}).get("total")
                if total is not None:
                    try:
                        genuine_totals.append(float(total))
                    except ValueError:
                        pass
            else:
                bboxes = []
                for bbox in r["label"].get("bboxes", []):
                    bboxes.append(bbox)
                if bboxes:
                    # key by image filename only
                    fname = Path(r["image_path"]).name
                    bbox_lookup[fname] = bboxes

        total_mean = float(np.mean(genuine_totals)) if genuine_totals else 0.0
        total_std = float(np.std(genuine_totals)) if genuine_totals else 1.0
        print(f"[train] total mean={total_mean:.2f}  std={total_std:.2f}")

        print("[train] Extracting image features...")
        backbone = build_model()
        X = _extract_features(backbone, img_paths)
        y = np.array(labels)

        print("[train] Training LogisticRegression...")
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        )
        clf.fit(X, y)
        print(f"[train] Done: forged={sum(labels)}  genuine={len(labels) - sum(labels)}")

        model_path = work_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "clf": clf,
                "total_mean": total_mean,
                "total_std": total_std,
                "bbox_lookup": bbox_lookup,
            }, f)
        print(f"[train] Saved to {model_path}")
        return str(work_dir)

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        """
        Run inference and write predictions to out_path.

        Args:
            model_dir: Path returned by train()
            data_dir:  Path to directory containing test.jsonl and images/
            out_path:  Path where predictions JSONL should be written
        """
        model_dir = Path(model_dir)
        data_dir = Path(data_dir)

        with open(model_dir / "model.pkl", "rb") as f:
            data = pickle.load(f)
        clf = data["clf"]
        total_mean = data["total_mean"]
        total_std = data["total_std"]

        test_jsonl = data_dir / "test.jsonl"
        if not test_jsonl.exists():
            candidates = list(data_dir.glob("*.jsonl"))
            if not candidates:
                raise FileNotFoundError(f"No .jsonl found in {data_dir}")
            test_jsonl = candidates[0]

        records = _load_jsonl(test_jsonl)
        print(f"[predict] {len(records)} records")

        img_paths = [data_dir / r["image_path"] for r in records]

        print("[predict] Extracting image features...")
        backbone = build_model()
        X = _extract_features(backbone, img_paths)

        preds = clf.predict(X)
        probas = clf.predict_proba(X)[:, 1]

        with open(out_path, "w", encoding="utf-8") as f:
            for r, pred, proba in zip(records, preds, probas):
                fields = r.get("fields", {})
                total = fields.get("total")

                outlier = False
                if total is not None:
                    try:
                        z = (float(total) - total_mean) / max(total_std, 1.0)
                        outlier = z > 3.0
                    except ValueError:
                        pass

                # final_forged = int(proba > 0.65)
                final_forged = int(int(pred) == 1 or outlier)

                out = {
                    "id": r["id"],
                    "vendor": fields.get("vendor"),
                    "date": fields.get("date"),
                    "total": total,
                    "is_forged": final_forged,
                }
                f.write(json.dumps(out) + "\n")

        print(f"[predict] Written to {out_path}")
