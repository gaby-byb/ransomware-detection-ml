# Machine Learning-based Ransomware Detection

This project is my Senior Capstone at the University of North Georgia.  
It aims to build a machine learning system that detects ransomware activity by analyzing system and network logs.

---

## 🚀 Project Overview

Ransomware remains one of the most disruptive forms of cybercrime, often leaving organizations with financial and operational damage.

The goal of this project is to:

- Train ML models on **static log data** to classify ransomware vs. benign activity.
- Normalize and preprocess logs for consistent feature extraction.
- Evaluate performance using multiple metrics (accuracy, precision, recall, F1-score).
- Compare results across different models (Decision Trees, Random Forests, SVM, etc.).

Tools & Libraries:

- Python 3.10+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn (for analysis)

---

## 📂 Repo Structure

data/ -> placeholder for datasets (not included in repo)
notebooks/ -> Jupyter notebooks for exploration and experiments
src/ -> source code (preprocessing, model training, utils)
reports/ -> notes, drafts, documentation

---

## ⚙️ Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/gaby-byb/ransomware-detection-ml.git
cd ransomware-detection-ml
pip install -r requirements.txt
```

## ✅ Progress

- [x] Repo initialized
- [ ] Add preprocessing pipeline
- [ ] Train baseline ML model
- [ ] Run evaluation metrics
- [ ] Write final report/paper

📌 Notes

    Datasets will not be pushed to this repo. See data/README.md for download instructions.

    This is a work in progress (target completion: Nov 15, 2025).

---

### 📝 `data/README.md`

# Data Notes

Datasets are not included in this repo due to size and security concerns.
Please obtain datasets from the following sources:

- [Kaggle Ransomware Detection Dataset](https://www.kaggle.com/)
- MITRE ATT&CK-based log samples

After downloading, place them in the `/data` folder.
