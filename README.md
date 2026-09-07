```markdown
# Post Analysis: Label Validation & Topic Classification

This repository provides a two‑part pipeline for analyzing social media or forum posts.

---

## Overview

The code is split into two main modules:

1. **Experiment** – a lightweight validation suite that checks the effectiveness of dataset labels for downstream tasks.
2. **Topic Classification** – a set of scripts that classify posts into predefined topic categories.

---

## 1. Experiment

The `experiment/` folder contains code to empirically verify how well the existing labels (e.g., sentiment, stance, or any categorical annotation) support a typical downstream task.

### What it does
- Runs a simple baseline model (e.g., logistic regression, SVM, or a small transformer) on the labeled dataset.
- Evaluates performance using metrics such as accuracy, F1‑score, and confusion matrix.
- Compares against random or constant baselines to confirm that the labels carry meaningful signal.
- Includes optional ablation tests (e.g., label noise injection) to stress‑test robustness.

### Key files
- `run_experiment.py` – main entry point for training and evaluation.
- `config_experiment.yaml` – configuration for model, splits, and metrics.
- `results/` – output folder with performance reports and plots.

### Usage
```bash
cd experiment
python run_experiment.py --config config_experiment.yaml
```

---

## 2. Topic Classification

The `topic_classification/` directory holds all code required to classify post texts into a set of user‑defined topic categories (e.g., *technology*, *politics*, *health*, *entertainment*).

### What it does
- Preprocesses raw text (tokenization, stopword removal, lemmatization).
- Trains a classifier (supports both traditional ML – TF‑IDF + linear models – and fine‑tuned transformer‑based models like BERT).
- Provides inference scripts to assign topics to new, unseen posts.
- Includes evaluation tools to measure per‑category precision/recall and to visualise the topic distribution.

### Key files
- `train_classifier.py` – trains the topic model on labeled data.
- `predict.py` – loads a trained model and predicts topics for new posts.
- `preprocess.py` – text cleaning and feature extraction utilities.
- `config_topic.yaml` – parameter file for model choice, hyperparameters, and data paths.
- `data/` – sample input data (training and test sets).

### Usage
```bash
# Train a new model
cd topic_classification
python train_classifier.py --config config_topic.yaml

# Predict on a CSV with a 'text' column
python predict.py --model path/to/model --input new_posts.csv --output predictions.csv
```

---

## Project Structure

```
.
├── experiment/
│   ├── run_experiment.py
│   ├── config_experiment.yaml
│   └── results/
├── topic_classification/
│   ├── train_classifier.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── config_topic.yaml
│   └── data/
├── requirements.txt
└── README.md
```

---

## Dependencies

All required packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

---

## Getting Started

1. Clone the repository.
2. Install dependencies.
3. Prepare your dataset (CSV format with a `text` column and, for training, a `label` column).
4. Run the experiment module to validate your labels.
5. Train and deploy the topic classifier on your posts.

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.
```
