import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pandas as pd

# === Config ===
MODEL_DIR = "model"  # path to your trained model
INPUT_CSV = "data/new_data.csv"  # your new data (optional)
OUTPUT_CSV = "data/predictions.csv"

# === Label mapping ===
label_map = {
    0: "barely-true",
    1: "false",
    2: "half-true",
    3: "mostly-true",
    4: "pants-fire",
    5: "true"
}

# === Load model and tokenizer ===
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # set to evaluation mode

# === Load new data ===
try:
    df = pd.read_csv(INPUT_CSV)
    texts = df["text"].tolist()
except FileNotFoundError:
    # fallback to sample texts if CSV not found
    texts = [
        "The government passed a new law today.",
        "Aliens have landed in New York!",
        "The COVID-19 vaccine is 100% effective."
    ]
    df = pd.DataFrame({"text": texts})

# === Tokenize input texts ===
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

# === Run inference ===
with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).tolist()

# === Map predictions back to labels ===
predicted_labels = [label_map[pred] for pred in predictions]

# === Save or print results ===
df["predicted_label"] = predicted_labels
df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")
for text, label in zip(texts, predicted_labels):
    print(f"Text: {text}\nPredicted label: {label}\n")
