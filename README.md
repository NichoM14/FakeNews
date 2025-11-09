# Fake News Detection

## Overview

An end-to-end **fake news detection pipeline** covering:

- Data collection, cleaning, and preprocessing
- Baseline models and transformer fine-tuning
- Model evaluation, explainability, and ethical considerations
- API serving and interactive demo (Streamlit + FastAPI)
- Reproducibility and MLOps (CI/CD, monitoring)

This project is **reproducible** and designed as a showcase of practical ML engineering.

## Project Structure

fake-news-detection/
├─ data/                     # raw and processed datasets
├─ notebooks/                # exploratory analysis & experiments
├─ src/
│  ├─ data/                  # data ingestion & preprocessing
│  ├─ models/                # model training scripts
│  ├─ eval/                  # evaluation scripts
│  ├─ serve/                 # FastAPI app
│  └─ utils/                 # shared utilities
├─ scripts/                  # helpers (download data, train, eval)
├─ demo/                     # Streamlit demo
├─ infra/                    # CI/CD templates, deployment manifests
├─ docs/                     # methodology, ethics, report
├─ README.md
└─ requirements.txt

## Setup

1. Install Python 3.10+
python --version

2. Install uv
pip install uv
uv --version

3. Create & activate project environment
cd /path/to/project
uv create project_env
uv activate project_env
uv install -r requirements.txt

## Usage

Download and prepare data
python scripts/download_data.py
python src/data/prepare_dataset.py

Train models
python src/models/baselines.py
python src/models/transformer_train.py

Evaluate models
python scripts/eval_pipeline.py

Serve model and run demo
python src/serve/app.py
streamlit run demo/streamlit_app.py

## Documentation

- EDA & preprocessing: notebooks/00_EDA.ipynb
- Baseline & transformer experiments: notebooks/01_baselines.ipynb, 02_transformers.ipynb
- Robustness & error analysis: notebooks/03_robustness_and_errors.ipynb
- Ethics & explainability: docs/ethics.md
- MLOps & CI/CD: docs/mlops.md
- Project report: docs/report.pdf

## Reproducibility

To fully reproduce all steps (baseline → transformer → evaluation → demo):
uv create project_env
uv activate project_env
uv install -r requirements.txt
bash scripts/run_full_experiment.sh

## Contributing

- Create a branch → PR → code review → merge
- Ensure tests pass before submitting a PR

## License

MIT License. See LICENSE for details.
