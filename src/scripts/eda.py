import csv
import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import utils


PROCESSED_DIR = "data/processed"

train_processed = pd.read_csv(os.path.join(PROCESSED_DIR, "train.tsv"), sep='\t', engine='python')

# Check missing values
print("Missing values in processed train.tsv:")
print(train_processed.isnull().sum())

# Countplot for label distribution
# plt.figure(figsize=(10,6))
# sns.countplot(data=train_processed, x="1")
# plt.show()

# top_states = train_processed["6"].value_counts().nlargest(10).index
# train_filtered = train_processed[train_processed["6"].isin(top_states)]

# plt.figure(figsize=(12,6))
# sns.countplot(data = train_filtered, x = "6", hue = "1")
# plt.xticks(rotation=90)
# plt.show()

# top_titles = train_processed["5"].value_counts().nlargest(10).index
# train_filtered2 = train_processed[train_processed["5"].isin(top_titles)]

# plt.figure(figsize=(12,6))
# sns.countplot(data = train_filtered2, x = "5", hue = "1")
# plt.xticks(rotation=90)
# plt.show()


# Read your TSV
df = train_processed

# # Split the CSV column and explode
# df_exploded = df.assign(values=df['3'].str.split(',')).explode('values')

# # Clean the values (remove whitespace)
# df_exploded['values'] = df_exploded['values'].str.strip()

# # Count occurrences
# n_top = 20  # Adjust this number
# value_counts = df_exploded['values'].value_counts().head(n_top).index

# df_filtered = df_exploded[df_exploded['values'].isin(value_counts)]
# value_label_counts = pd.crosstab(df_filtered['values'], df_filtered['1'])

# value_label_counts.plot(kind='bar', stacked=False)
# plt.title(f'Top {10} Most Frequent Values')
# plt.xlabel('Values')
# plt.ylabel('Count')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

#utils.clustering_goat(df, 2)

num_unique_speakers = df['4'].nunique()
avg_statements = len(df) / num_unique_speakers
print(f"Number of unique speakers: {num_unique_speakers}")
print(f"Average number of statements per speaker: {avg_statements:.2f}")

# Exploring statement length with label correlation
# df["text_len"] = df["2"].str.len()

# plt.figure(figsize=(10,5))
# sns.histplot(df["text_len"], bins=40)
# plt.title("Statement Length Distribution")
# plt.show()
# plt.figure(figsize=(12,6))
# sns.boxplot(x=df["1"], y=df["text_len"])
# plt.title("Statement Length by Label")
# plt.xticks(rotation=45)
# plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.iloc[:,8:13].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation of Credit History Columns")
plt.show()

