# 🛡️ Financial Transaction Fraud Detection

This project builds a Machine Learning classification model using **Random Forest** to detect fraudulent financial transactions (`is_fraud`) from the Kaggle **Playground Series - Season 6, Episode 11** dataset.

---

## 📌 Project Overview

- **Goal:** Predict whether a financial transaction is fraudulent (`is_fraud`) using transaction, card, device, and behavioral features.
- **Dataset:**
  - **Training set:** `699,960` rows, `26` columns
  - **Test set:** `299,984` rows, `25` columns
- **Model Used:** `RandomForestClassifier` (`n_estimators=100`, `max_depth=12`, `random_state=42`, `n_jobs=-1`)

---

## 🛠️ Tech Stack & Libraries

- **Python** (`3.13.18`)
- **Pandas & NumPy** – Data loading and manipulation
- **Scikit-learn** – `OrdinalEncoder`, `RandomForestClassifier`, evaluation metrics
- **Matplotlib & Seaborn** – Data visualization

---

## ⚙️ Workflow

1. **Load Data:** Read `train.csv` and `test.csv` from `/kaggle/input/competitions/playground-series-s6e11/`.
2. **Preprocessing:** Encode categorical columns using `OrdinalEncoder` (handling unseen values with `unknown_value=-1`).
3. **Feature & Target Split:** Separate features (`X`) and target (`y = is_fraud`).
4. **Model Training:** Train a `RandomForestClassifier` on the training dataset.
5. **Prediction & Submission:** Generate predictions on `test.csv` and save the output to `submission.csv` with `id` and `is_fraud` columns.

---

## 📂 Repository Structure

```text
├── notebook.ipynb      # Kaggle notebook with data preprocessing, training & inference
├── submission.csv      # Generated predictions file (id, is_fraud)
└── README.md           # Project documentation
