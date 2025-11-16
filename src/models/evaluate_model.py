# src/models/evaluate_model.py

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# === Config ===
MODEL_DIR = "model"
VAL_CSV = "data/processed/val.csv"  # adjust if your validation file is named differently
OUT_CSV = "data/predictions.csv"
BATCH_SIZE = 16
MAX_LENGTH = 256  # max tokens per text

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load model and tokenizer ===
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# === Load validation data ===
df = pd.read_csv(VAL_CSV)
if 'text' not in df.columns or 'label' not in df.columns:
    raise SystemExit("Validation CSV must have 'text' and 'label' columns")

texts = df['text'].astype(str).tolist()
true_labels_raw = df['label'].tolist()

# === Map string labels to integers ===
if any(isinstance(l, str) for l in true_labels_raw):
    unique_labels = sorted(set(true_labels_raw))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    true_labels = [label_map[l] for l in true_labels_raw]
    print("Mapped string labels to ints:", label_map)
else:
    true_labels = [int(l) for l in true_labels_raw]

# === Batch prediction ===
preds = []
confidences = []

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    # Move to correct device
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
        batch_conf = torch.softmax(logits, dim=1).max(dim=1).values.cpu().tolist()

    preds.extend(batch_preds)
    confidences.extend(batch_conf)

# === Metrics ===
accuracy = accuracy_score(true_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')
cm = confusion_matrix(true_labels, preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print("Confusion matrix:")
print(cm)

# === Save predictions with confidence ===
df['pred_label_id'] = preds
df['pred_confidence'] = confidences
df.to_csv(OUT_CSV, index=False)
print(f"Predictions saved to {OUT_CSV}")
