# Financial Fraud Detection Using Machine Learning

An end-to-end machine learning project for detecting fraudulent financial transactions using the IEEE-CIS Fraud Detection dataset. The project compares Logistic Regression, Random Forest, and XGBoost, then improves the final system through probability calibration, threshold tuning, and SHAP-based interpretability.

The pipeline is designed for highly imbalanced tabular fraud data and emphasizes three things: leakage-safe preprocessing, recall-oriented model optimization, and explainable predictions.

## Problem Statement

The task is binary classification: predict whether a transaction is fraudulent (`isFraud = 1`) or legitimate (`isFraud = 0`). The dataset is difficult because it is highly imbalanced, time-ordered, high-dimensional after merging transaction and identity tables, and sensitive to temporal leakage if split incorrectly.

The report states that only about 3.5% of transactions are fraudulent, which makes accuracy alone misleading and shifts focus toward recall, PR-AUC, and cost-sensitive evaluation.

## Dataset

This project uses the IEEE-CIS Fraud Detection dataset released by Vesta Corporation for the 2019 Kaggle competition.

Key details from the report:

- 590,540 transaction rows.
- 394 transaction features and 41 identity features before filtering.
- Join key: `TransactionID`.
- Time column: `TransactionDT`.
- Target column: `isFraud`.

## Pipeline

The implementation follows a modular chronological pipeline described in the report.

1. Load `train_transaction.csv` and `train_identity.csv`.
2. Merge both tables using `TransactionID`.
3. Drop columns with more than 55% missing values.
4. Sort by `TransactionDT` and split chronologically into 80% train, 10% calibration, and 10% test.
5. Perform leakage-safe feature engineering and imputation using training-set statistics only.
6. Apply label encoding with unseen-value handling.
7. Remove near-zero-variance features.
8. Train baseline and advanced models.
9. Calibrate probabilities using isotonic regression.
10. Select the final operating threshold using the Precision-Recall curve and F2 score.
11. Evaluate on the held-out test set.
12. Use SHAP for global and local interpretability.

## Feature Engineering

The report highlights several custom fraud-oriented features that improved model quality.

- `logTransactionAmt`: log transform of transaction amount.
- `hour`, `dayofweek`, `isnight`: temporal behavior indicators.
- `card1meanamt`, `card1stdamt`, `amtToMean`, `amtDev`: personalized spending anomaly features.
- `timeSinceLast`, `card1TxnVelocity`, `uidCount`: velocity and repeat-use indicators.
- `cardAddr`, `uid`, `PREmailMatch`, `D1NullFlag`: identity-consistency and missingness signals.

The report notes that five of the top ten SHAP-important features were engineered by the team, supporting the value of domain-specific feature creation.

## Models

### 1) Logistic Regression

Used as the baseline to quantify the effect of extreme class imbalance and test linear separability.

Reported results:

- Recall: 0.35.
- Precision: 0.62.
- PR-AUC: 0.79.
- ROC-AUC: 0.79.

### 2) Random Forest

Used as the intermediate non-linear model to capture interactions missed by Logistic Regression.

Reported results:

- Recall: 0.68.
- Precision: 0.73.
- PR-AUC: 0.88.
- ROC-AUC: 0.91.
- Training time: about 50 minutes, with roughly 14 GB RAM usage.

### 3) XGBoost

Selected as the final model because of regularization, efficient histogram-based tree building, early stopping, and strong performance on imbalanced tabular data.

Reported results:

- Default threshold recall: 0.72.
- Default threshold precision: 0.81.
- PR-AUC: 0.93-0.95.
- ROC-AUC: 0.96.
- Training time: about 10 minutes.

## Threshold Tuning

Instead of using the default 0.5 classification threshold, the final system calibrates probabilities with isotonic regression and then selects an F2-optimal threshold from the Precision-Recall curve.

The report identifies an optimal threshold near 0.18, producing about 0.90-0.91 recall with precision in the 0.55-0.65 range.

This threshold choice is justified because missing a fraud case is significantly costlier than incorrectly flagging a legitimate transaction.

## Final Results

At the selected operating point, the system achieves strong fraud recall while remaining practical for analyst review workflows.

Key reported outcomes:

- Fraud recall: about 0.90-0.91.
- Fraud precision: about 0.57-0.65.
- F1-score (fraud class): about 0.70.
- PR-AUC: about 0.93-0.95.
- ROC-AUC: about 0.96.

The report also emphasizes that threshold optimization improved recall almost as much as changing model family, which is an important practical takeaway.

## Interpretability

SHAP was used for both global and local explanation of the final XGBoost model.

The most influential signals included `amtToMean`, `logTransactionAmt`, `uidMeanAmt`, `card1MeanAmt`, `timeSinceLast`, `card1TxnVelocity`, and `isnight`.

These explanations helped verify that the model learned meaningful fraud patterns such as spending anomalies, unusual velocity, and suspicious transaction timing rather than relying on spurious shortcuts.

## Suggested Repository Structure

```text
fraud-detection-ml/
├── README.md
├── requirements.txt
├──.gitignore
├── configs/
│  └── config.yaml
├── data/
│  ├── raw/
│  │  ├── train_transaction.csv
│  │  └── train_identity.csv
│  ├── processed/
│  └── interim/
├── notebooks/
│  ├── 01_eda.ipynb
│  ├── 02_feature_engineering.ipynb
│  ├── 03_model_baseline_lr.ipynb
│  ├── 04_model_random_forest.ipynb
│  ├── 05_model_xgboost.ipynb
│  ├── 06_threshold_tuning.ipynb
│  └── 07_shap_analysis.ipynb
├── src/
│  ├── __init__.py
│  ├── data_loader.py
│  ├── preprocess.py
│  ├── features.py
│  ├── split.py
│  ├── encode.py
│  ├── train_baseline.py
│  ├── train_random_forest.py
│  ├── train_xgboost.py
│  ├── calibrate.py
│  ├── threshold.py
│  ├── evaluate.py
│  ├── explain.py
│  └── utils.py
├── artifacts/
│  ├── models/
│  ├── metrics/
│  ├── figures/
│  └── shap/
└── docs/
  └── report-assets/
```

## Code File Order

A good order for writing or organizing the codebase is:

1. `data_loader.py` - load and merge transaction and identity files.
2. `split.py` - sort by `TransactionDT` and create chronological train/calibration/test splits.
3. `preprocess.py` - null filtering, imputation, type handling, and column management.
4. `features.py` - domain-specific feature engineering with train-only fit statistics.
5. `encode.py` - label encoding with unseen-category handling.
6. `train_baseline.py` - Logistic Regression baseline.
7. `train_random_forest.py` - Random Forest benchmark.
8. `train_xgboost.py` - two-stage XGBoost training and top-feature selection.
9. `calibrate.py` - isotonic probability calibration.
10. `threshold.py` - Precision-Recall analysis and F2-based threshold selection.
11. `evaluate.py` - classification report, confusion matrix, PR-AUC, ROC-AUC.
12. `explain.py` - SHAP global and local explanations.
13. `utils.py` - shared helpers for saving artifacts, logging, and reproducibility.

## Minimal Starter Files

### `requirements.txt`

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=3.2.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
pyyaml>=6.0
joblib>=1.3
```

### `.gitignore`

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.venv/
venv/
data/raw/
data/interim/
data/processed/
artifacts/models/
artifacts/metrics/
artifacts/figures/
artifacts/shap/
.DS_Store
```

## Example Run Order

```bash
python -m src.data_loader
python -m src.split
python -m src.preprocess
python -m src.features
python -m src.encode
python -m src.train_baseline
python -m src.train_random_forest
python -m src.train_xgboost
python -m src.calibrate
python -m src.threshold
python -m src.evaluate
python -m src.explain
```

## Notes

- This project is designed as an offline batch fraud detection pipeline, not a real-time production fraud API.
- The report explicitly lists real-time inference, graph-based methods, drift monitoring, federated learning, and an explainability dashboard as future enhancements.
- If you publish this repository, do not upload the Kaggle dataset files directly; provide setup instructions instead.

