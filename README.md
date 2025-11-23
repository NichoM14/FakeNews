# FakeNews

## Setup

```
uv add transformers datasets
uv add torch torchvision torchaudio --extra-index-url  https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

## Download the data

`uv run download_data.py`

## To Do List
- [X] Setup the Project
- [ ] Clean valid and test in `eda.py`
- [ ] Split `eda.py` into 2 files: One for processed data and other for EDA