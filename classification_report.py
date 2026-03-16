import json
from sklearn.metrics import classification_report

labels = {}
with open("data/test/labels.jsonl") as f:
    for line in f:
        r = json.loads(line)
        labels[r["id"]] = r["label"]["is_forged"]

preds = {}
with open("tmp_work/predictions.jsonl") as f:
    for line in f:
        r = json.loads(line)
        preds[r["id"]] = r["is_forged"]

ids = sorted(labels.keys())
y_true = [labels[i] for i in ids]
y_pred = [preds[i]  for i in ids]

print(f"Total: {len(y_true)}")
print(f"Authentic: {y_true.count(0)}")
print(f"Forged: {y_true.count(1)}")

print(f"Predicted forged: {y_pred.count(1)}")
print(f"Predicted authentic: {y_pred.count(0)}")

print(classification_report(y_true, y_pred, target_names=["authentic", "forged"]))