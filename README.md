# Financial Fraud Detection using Machine Learning

This project builds a fraud detection pipeline on the IEEE-CIS dataset using the two source files `train_transaction.csv` and `train_identity.csv`. The implementation is contained in a single `main.py` script that loads the data, engineers fraud-oriented features, trains an XGBoost model, calibrates predicted probabilities, chooses an operating threshold, evaluates performance, and explains predictions with SHAP.

## Project Overview

The goal is to classify each transaction as fraudulent (`isFraud = 1`) or legitimate (`isFraud = 0`). The pipeline is designed for a highly imbalanced dataset, so it focuses more on recall, PR-AUC, and practical fraud-catching ability than on raw accuracy.

Unlike a multi-file production repository, this version is a code-complete research project centered around one main script. The workflow is still modular inside that script, with distinct sections for loading, preprocessing, feature engineering, training, calibration, evaluation, and explainability.

## Dataset Used

The project uses these two CSV files as input:

- `train_transaction.csv`
- `train_identity.csv`

These are merged using `TransactionID` with a left join, so all transaction rows are preserved even when identity information is missing.

## What `main.py` does

The `main.py` file performs the full end-to-end workflow:

1. Loads `train_transaction.csv` and `train_identity.csv`.
2. Merges them on `TransactionID`.
3. Drops columns with more than 55% missing values.
4. Fills numeric nulls with median values and categorical nulls with `"missing"`.
5. Sorts records chronologically using `TransactionDT`.
6. Splits the data into 80% train, 10% calibration, and 10% test.
7. Engineers domain-specific fraud features.
8. Encodes categorical columns using `LabelEncoder`.
9. Drops near-zero variance numeric columns.
10. Trains an XGBoost classifier with fraud-weighted samples.
11. Calibrates probabilities using isotonic regression.
12. Selects the best threshold from the precision-recall tradeoff.
13. Evaluates on the test set using classification metrics and confusion matrix.
14. Generates SHAP-based feature importance and local explanation plots.

## Feature Engineering

A major part of the project is custom feature engineering. The script creates behavior-based and anomaly-based features such as:

- `logTransactionAmt`
- `amtIsRound`
- `amtIs1Dollar`
- `hour`, `dayOfWeek`, `isWeekend`, `isNight`, `isBusinessHour`
- cyclical time features like `hourSin`, `hourCos`, `dowSin`, `dowCos`
- per-card spending statistics like mean, std, count, and median
- deviation features such as `amtToMean`, `amtDev`, `amtZScore`
- identity combination features like card-address combinations
- transaction velocity features such as `timeSinceLast`, `logTimeSinceLast`, `isRapidTxn`, `card1Count1h`, `card1Count24h`, `card1TxnVelocity`
- high-amount flags like `isHighAmt` and `isExtremeAmt`
- email match and frequency features
- product, billing address, device, and fraud-rate encoded features
- anonymized feature standardization for `C*` and `D*` columns
- mismatch summary features for `M*` columns

These features are fitted on training data and then applied forward to calibration and test data to reduce leakage.

## Model Used

The final model in `main.py` is `XGBClassifier` from XGBoost with settings tuned for imbalanced fraud detection. The script uses:

- `n_estimators=3000`
- `learning_rate=0.02`
- `max_depth=6`
- `subsample=0.80`
- `colsample_bytree=0.70`
- `colsample_bylevel=0.70`
- `colsample_bynode=0.70`
- `reg_alpha=2`
- `reg_lambda=2`
- `min_child_weight=3`
- `gamma=0.1`
- `eval_metric='aucpr'`
- `tree_method='hist'`
- `device='cuda'`
- `early_stopping_rounds=100`

Fraud samples are upweighted using custom sample weights, with fraud transactions receiving a weight multiplier of `25.0`.

## Probability Calibration

After model training, the script calibrates prediction probabilities using `CalibratedClassifierCV` with:

- `method='isotonic'`
- `cv='prefit'`

This step is important because weighted training improves recall but can distort raw probability estimates.

## Threshold Selection

The model does not rely on the default threshold of `0.5`. Instead, it computes a precision-recall curve on the calibration split and selects a threshold using an F2-style objective, where recall matters more than precision.

The code also applies a target precision preference (`TARGET_PRECISION = 0.80`) and prints threshold tradeoffs so the operating point can be adjusted depending on business needs.

## Evaluation

The script evaluates the chosen threshold on the test set using:

- classification report
- confusion matrix
- ROC-AUC
- PR-AUC
- fraud alert precision
- fraud catch rate / recall

It also prints a practical interpretation, such as how many of the flagged transactions are actually fraud and how many fraud cases are being caught.

## Explainability

The project uses SHAP (`shap.TreeExplainer`) for interpretability. The script generates:

- SHAP summary bar plot
- SHAP summary dot plot
- SHAP force plot for an individual prediction

This helps explain which features are pushing predictions toward fraud or non-fraud.

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- SHAP

## File Structure

```text
.
├── main.py
├── train_transaction.csv
├── train_identity.csv
└── README.md
```

## How to Run

1. Place `main.py`, `train_transaction.csv`, and `train_identity.csv` in the same folder.
2. Install the required packages.
3. Run the script.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
python main.py
```

## Expected Output

Running the script will produce:

- dataset shape and fraud-rate logs
- train / calibration / test split information
- XGBoost training progress
- threshold analysis output
- final classification metrics
- confusion matrix visualization
- precision-recall visualization
- SHAP plots for global and local interpretation

## Notes

- The current implementation is built as an offline batch fraud detection workflow.
- It is designed as a single-script project rather than a production API or package.
- The script expects GPU support through `device='cuda'`; if GPU is unavailable, this can be changed to CPU.
- Because the project uses chronological splitting and train-only fitted encoders/statistics, it is more realistic than a random-split notebook workflow.

## Future Improvements

Some strong next steps for this project would be:

- split `main.py` into reusable modules
- save trained models and encoders as artifacts
- add a `requirements.txt`
- add command-line arguments or config support
- build a Streamlit dashboard for fraud analysis
- create a real-time API for scoring new transactions
- add experiment tracking and model versioning

## Author Note

This README is written to match the actual code structure of the project, where the full pipeline is implemented in `main.py` and the data sources are `train_transaction.csv` and `train_identity.csv`.
