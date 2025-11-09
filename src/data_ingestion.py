import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV/TSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, sep="\t", header=0)  # use sep="\t" for LIAR
    print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        file_path = f"data/raw/liar-dataset/{split}.tsv"
        df = load_data(file_path)
        print(df.head())
