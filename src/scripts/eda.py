import csv
import os
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
train = pd.read_csv(os.path.join(RAW_DIR, "train.tsv"), sep ='\t', header = None, usecols=range(14), names=range(14), quoting=csv.QUOTE_NONE)
valid = pd.read_csv(os.path.join(RAW_DIR, "valid.tsv"), sep ='\t', header = None)
test = pd.read_csv(os.path.join(RAW_DIR, "test.tsv"), sep ='\t', header = None)

print("Train:", train.shape)
print("Valid:", valid.shape)
print("Test:", test.shape)
print(train.head())

print("\nMissing Values:", train.isnull().sum())

train.fillna({5: "unknown speaker's job title"}, inplace=True)
train.fillna({6: "unknown state info"}, inplace=True)
train.fillna({13: "unknown context"}, inplace=True)
train.to_csv(os.path.join(PROCESSED_DIR, "train.tsv"), sep ="\t", index=False, quoting=csv.QUOTE_ALL)

train_processed = pd.read_csv(os.path.join(PROCESSED_DIR, "train.tsv"), sep='\t', engine='python')

# Check missing values
print("Missing values in processed train.tsv:")
print(train_processed.isnull().sum())