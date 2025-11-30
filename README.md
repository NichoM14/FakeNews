# FakeNews

## Setup

```
uv add transformers datasets
uv add torch torchvision torchaudio --extra-index-url  https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
uv add sentence-transformers hdbscan
uv add umap-learn
```

## Download the data

`uv run download_data.py`

## Clean and process the Data

`uv run src/scripts/eda.py`

## To Do List
- [X] Setup the Project
- [X] Clean valid and test in `clean_data.py`
- [X] Split `clean_data.py` into 2 files: One for processed data and other for EDA

## EDA

Remove comments in `eda.py` to observe the data. HDBSCAN on statement column doesnt give much.