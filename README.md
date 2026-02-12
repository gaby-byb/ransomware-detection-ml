# Static PE-Header Based Ransomware Detection

This repository implements an end-to-end machine learning pipeline for detecting ransomware using static Portable Executable (PE) header features.

Rather than evaluating a single dataset in isolation, this project systematically analyzes how static PE-header features perform across three progressively difficult binary classification tasks, exposing both strengths and limitations of static-only detection.

---

## Project Overview

This project asks the question:
Do static PE-header features generalize across different classification tasks and malware distributions?

To answer this, three distinct binary tasks were constructed:

- Ransomware vs Benign
- Malware vs Benign
- Ransomware vs Other Malware Families

## Methodology

The pipeline is fully reproducible and includes:

### Data Preprocessing

- Removal of identifier and leakage-prone features
- Log transformation of skewed numeric fields
- One-hot encoding of categorical fields (e.g., Machine type)
- Missing value handling
- Feature alignment to match training schema

### Class Imbalance Handling

- SMOTE (Synthetic Minority Oversampling Technique)
- Model-level class weighting
- Stratified 5-fold cross-validation

### Models Evaluated

- Random Forest
- XGBoost
- Linear SVM

Hyperparameters were tuned using GridSearchCV or RandomizedSearchCV.

### Evaluation Metrics

Due to dataset imbalance, evaluation emphasizes:

- Matthews Correlation Coefficient (MCC)
- Precision / Recall / F1-score
- PR-AUC
- Confusion Matrix

### Model Interpretability

SHAP (SHapley Additive Explanations) was applied to tree-based models to:

- Identify globally important PE-header features
- Analyze feature contribution distributions
- Compare feature influence across datasets

---

## Key Findings

- Static PE-header features perform strongly in ransomware vs benign and malware vs benign tasks.
- Performance degrades in ransomware vs other malware families, revealing limitations of static-only detection.
- No single feature dominates predictions; decisions rely on feature combinations.
- Tree-based models provided the strongest balance of robustness and interpretability.

These results indicate that static analysis alone is insufficient for fine-grained malware family discrimination.

---

## Repository Structure

data/ -> Dataset placeholders (not included)
notebooks/ -> Exploratory analysis and experiments
src/ -> Preprocessing, training, evaluation pipeline
reports/ -> Figures, SHAP plots, manuscript drafts

---

## Environment

- Python 3.11
- scikit-learn
- XGBoost
- imbalanced-learn
- SHAP
- pandas
- numpy
- matplotlib

See requirements.txt for exact versions.

---

## ⚙️ Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/gaby-byb/ransomware-detection-ml.git
cd ransomware-detection-ml
pip install -r requirements.txt
```

📌 Notes

    Datasets will not be pushed to this repo. See data/README.md for download instructions.

    This is a work in progress (target completion: Nov 15, 2025).

---

### 📝 `data/README.md`

# Data Notes

Datasets are not included in this repo due to size and security concerns.
Please obtain datasets from the following sources:

- [ransomware_vs_benign dataset](https://www.kaggle.com/datasets/amdj3dax/ransomware-detection-data-set)
- [malware_vs_benign dataset](https://www.kaggle.com/datasets/fanbyprinciple/file-pe-headers)
- [ransomware_vs_malware](https://github.com/DA-Proj/PE-Malware-Dataset1/blob/main/Header.csv)

After downloading, place them in the `/data` folder.
