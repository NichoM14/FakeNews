import csv
import os
import pandas as pd


def remove_null_value(df):
    df.fillna({5: "unknown speaker's job title"}, inplace=True)
    df.fillna({6: "unknown state info"}, inplace=True)
    df.fillna({13: "unknown context"}, inplace=True)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
train = pd.read_csv(os.path.join(RAW_DIR, "train.tsv"), sep ='\t', header = None, usecols=range(14), names=range(14), quoting=csv.QUOTE_NONE)
valid = pd.read_csv(os.path.join(RAW_DIR, "valid.tsv"), sep ='\t', header = None, usecols=range(14), names=range(14), quoting=csv.QUOTE_NONE)
test = pd.read_csv(os.path.join(RAW_DIR, "test.tsv"), sep ='\t', header = None, usecols=range(14), names=range(14), quoting=csv.QUOTE_NONE)

print("Train:", train.shape)
print("Valid:", valid.shape)
print("Test:", test.shape)
print(train.head())
print(valid.head())
print(test.head())

print("\nMissing Values:", train.isnull().sum())
print("\nMissing Values:", valid.isnull().sum())
print("\nMissing Values:", test.isnull().sum())

remove_null_value(train)
remove_null_value(valid)
remove_null_value(test)

train.to_csv(os.path.join(PROCESSED_DIR, "train.tsv"), sep ="\t", index=False, quoting=csv.QUOTE_ALL)
valid.to_csv(os.path.join(PROCESSED_DIR, "valid.tsv"), sep ="\t", index=False, quoting=csv.QUOTE_ALL)
test.to_csv(os.path.join(PROCESSED_DIR, "test.tsv"), sep ="\t", index=False, quoting=csv.QUOTE_ALL)

train_processed = pd.read_csv(os.path.join(PROCESSED_DIR, "train.tsv"), sep='\t', engine='python')
valid_processed = pd.read_csv(os.path.join(PROCESSED_DIR, "valid.tsv"), sep='\t', engine='python')
test_processed = pd.read_csv(os.path.join(PROCESSED_DIR, "test.tsv"), sep='\t', engine='python')

# Check missing values
print("Missing values in processed train.tsv:")
print(train_processed.isnull().sum())
print(valid_processed.isnull().sum())
print(test_processed.isnull().sum())

