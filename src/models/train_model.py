# src\models\train_model.py

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# === Config ===
CSV_PATH = "data/processed/train.csv"  # Update if your file name is different
MODEL_DIR = "model"
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2  # binary classification

# === Load dataset ===
# === Load dataset ===
df = pd.read_csv(CSV_PATH)
df = df.sample(n=200, random_state=42)  # <- only 200 rows for quick test
print(f"Loaded {len(df)} rows from {CSV_PATH}")


texts = df['text'].tolist()       # ignore 'id'
# === Labels ===
labels = df['label'].tolist()
label_mapping = {label: i for i, label in enumerate(sorted(set(labels)))}
print("Label mapping:", label_mapping)
labels = [label_mapping[label] for label in labels]
NUM_LABELS = len(label_mapping)


# Train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels
)

# === Tokenization ===
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# === Torch dataset ===
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# === Model ===
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=1,          # only 1 epoch for testing
    per_device_train_batch_size=4,  # smaller batch to avoid memory issues
    per_device_eval_batch_size=4,
    evaluation_strategy="no",     # skip evaluation for speed
    save_strategy="no",           # skip saving for now
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
)


# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# === Train ===
trainer.train()

# === Save model ===
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print(f"Training complete. Model saved to {MODEL_DIR}")
