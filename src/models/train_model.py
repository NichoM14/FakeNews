# src/models/train_model.py

import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# === Config ===
CSV_PATH = "data/processed/train.csv"
MODEL_DIR = "model"
MODEL_NAME = "distilbert-base-uncased"

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load dataset ===
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

# === Label mapping ===
label_map = {
    "barely-true": 0,
    "false": 1,
    "half-true": 2,
    "mostly-true": 3,
    "pants-fire": 4,
    "true": 5
}
df["label"] = df["label"].map(label_map)
labels = df["label"].tolist()
NUM_LABELS = len(label_map)
texts = df["text"].tolist()
print("Label mapping:", label_map)

# === Train/validation split ===
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.1,
    random_state=42,
    stratify=labels
)

# === Tokenization ===
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# === Torch dataset ===
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# === Compute class weights ===
label_counts = np.bincount(train_labels)
weights = 1.0 / label_counts
weights = torch.tensor(weights, dtype=torch.float).to(device)
print("Class weights:", weights)

# === Load model ===
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)
model.to(device)  # make sure model is on correct device

# === Metrics function ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# === Weighted Trainer ===
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Move labels to same device
        labels = inputs.get("labels").to(device)
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=3e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True if torch.cuda.is_available() else False
)

# === Trainer ===
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# === Train ===
print("Starting training...")
trainer.train()

# === Save model ===
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Training complete. Model saved to {MODEL_DIR}")
