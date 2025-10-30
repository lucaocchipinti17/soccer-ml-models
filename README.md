# ⚽ soccer-ml-models-main

End-to-end experiments and utilities for **soccer match outcome/score prediction**.

---

## 📦 What’s Inside

- Web scraping utilities to collect historical match data.
- Data cleaning & feature engineering pipeline.
- Training scripts for one or more models.

### Algorithms / Models
- Custom multi-layer neural network implemented in NumPy (forward/backprop).

---

## 🗂️ Repository Structure (truncated)
```
./
  main.py
  data/
    clean.py
    normalized_data.csv
    scraper.py
    train_prep.py
  models/
    nn_diff.py
    nn_goals.py
```

---

## ⚙️ Setup

```bash
# 1) Clone
git clone https://github.com/lucaocchipinti17/soccer-ml-models-main.git
cd soccer-ml-models-main

# 2) Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

---

## ▶️ Usage (examples)

> Adjust paths and script names to match your repository layout.

- **Preprocess data**
```bash
python src/data_preprocessing.py
```

- **Train a model**
```bash
python src/train.py
```

- **Evaluate**
```bash
python src/evaluate_models.py
```

- **Run a notebook**
```bash
jupyter lab
```

---

## 📝 Notes

- Data directories detected: data
- Notebooks detected: none
- CSV samples: data/normalized_data.csv

---

## 🛣️ To-Do / Future Work

- [ ] Review and document all CLI entry points and parameters
- [ ] Add a reproducible data acquisition script (scrape or download)
- [ ] Formalize a Makefile / task runner for end-to-end pipeline
- [ ] Unit tests for preprocessing and metrics
- [ ] Add experiment tracking (Weights & Biases or MLflow)
- [ ] Provide baseline metrics with fixed train/valid/test splits
- [ ] Dockerfile for environment reproducibility

---

## 👤 Author

Luca Occhipinti