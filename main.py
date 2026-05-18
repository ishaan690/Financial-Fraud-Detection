import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)

TRANSACTION_PATH = "/train_transaction.csv" 
IDENTITY_PATH    = "/train_identity.csv"    


# Step 1: Load Data
# low_memory=False avoids mixed dtype warnings on the IEEE dataset
# the safe_read trick converts object columns that are secretly numeric

def safe_read_csv(path):
    df = pd.read_csv(path, low_memory=False)
    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors='ignore')
    return df

print("Loading data...")
train_transaction = safe_read_csv(TRANSACTION_PATH)
train_identity    = safe_read_csv(IDENTITY_PATH)

# left join keeps all transactions even when identity info is missing
data = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
print(f"Shape after merge: {data.shape}")

del train_transaction, train_identity



# Step 2: Basic Cleaning
# 55% threshold is a reasonable cutoff - keeps useful columns like C/D/M series
# which have lots of NaNs but carry strong fraud signal

data = data.loc[:, data.isnull().mean() < 0.55]

num_cols      = data.select_dtypes(include=[np.number]).columns
cat_cols_init = data.select_dtypes(include='object').columns

# median imputation is more robust than mean for skewed financial data
data[num_cols]      = data[num_cols].fillna(data[num_cols].median())
data[cat_cols_init] = data[cat_cols_init].fillna("missing")

if 'isFraud' not in data.columns:
    raise KeyError("isFraud column not found - check if merge worked correctly")

print(f"Fraud rate: {data['isFraud'].mean():.4f}  ({data['isFraud'].sum()} fraud out of {len(data)} total)")



# Step 3: Time-Based Train / Cal / Test Split
# sorting by TransactionDT is critical - random split would leak future
# data into training, which inflates metrics and fails in production

data = data.sort_values('TransactionDT').reset_index(drop=True)

n         = len(data)
train_end = int(n * 0.80)
cal_end   = int(n * 0.90)

train_data = data.iloc[:train_end].copy()
cal_data   = data.iloc[train_end:cal_end].copy()
test_data  = data.iloc[cal_end:].copy()

del data

print(f"\nTrain : {train_data.shape}  fraud={train_data['isFraud'].mean():.4f}")
print(f"Cal   : {cal_data.shape}    fraud={cal_data['isFraud'].mean():.4f}")
print(f"Test  : {test_data.shape}   fraud={test_data['isFraud'].mean():.4f}")



# Step 4: Feature Engineering
# ALL aggregates are fitted on train_data only, then mapped to cal/test
# this is the most important thing to get right - any leakage here
# gives you fake good metrics that fall apart on new data

def engineer_features(df, ref_df=None, fit=True, encoders=None):
    df = df.copy()
    if encoders is None:
        encoders = {}

    #Transaction Amount Features
    # log transform handles the heavy right skew in transaction amounts
    if 'TransactionAmt' in df.columns:
        df['log_TransactionAmt'] = np.log1p(df['TransactionAmt'])
        df['amt_cents']          = (df['TransactionAmt'] * 100 % 100).round(0)
        df['amt_is_round']       = (df['amt_cents'] == 0).astype(np.int8)
        # round-number transactions are common in fraud (e.g., testing with $1.00)
        df['amt_is_1_dollar']    = (df['TransactionAmt'].round(2) == 1.00).astype(np.int8)

    #Time Features
    # TransactionDT is seconds elapsed from some reference point
    if 'TransactionDT' in df.columns:
        df['hour']        = (df['TransactionDT'] / 3600) % 24
        df['dayofweek']   = ((df['TransactionDT'] / (3600 * 24)) % 7).astype(int)
        df['is_weekend']  = (df['dayofweek'] >= 5).astype(np.int8)
        df['is_night']    = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(np.int8)
        df['is_business'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(np.int8)

        # cyclical encoding: hour 23 and hour 0 should be close in feature space
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin']  = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos']  = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # global fallback stats for unseen cards
    global_amt_mean = ref_df['TransactionAmt'].mean() if fit else encoders.get('global_amt_mean', 0)
    global_amt_std  = ref_df['TransactionAmt'].std()  if fit else encoders.get('global_amt_std', 1)
    if fit:
        encoders['global_amt_mean'] = global_amt_mean
        encoders['global_amt_std']  = global_amt_std

    #Per-Card Aggregates
    # card1 is the most important card identifier in this dataset
    for col in ['card1', 'card2', 'card3']:
        if col not in df.columns:
            continue
        if fit:
            agg = ref_df.groupby(col)['TransactionAmt'].agg(['mean', 'std', 'count', 'median'])
            agg.columns = [f'{col}_mean_amt', f'{col}_std_amt',
                           f'{col}_count',    f'{col}_med_amt']
            encoders[f'{col}_agg'] = agg

        agg = encoders[f'{col}_agg']
        df  = df.merge(agg, left_on=col, right_index=True, how='left')
        df[f'{col}_mean_amt'].fillna(global_amt_mean, inplace=True)
        df[f'{col}_std_amt'].fillna(0,                inplace=True)
        df[f'{col}_count'].fillna(1,                  inplace=True)
        df[f'{col}_med_amt'].fillna(global_amt_mean,  inplace=True)

    # how much does this transaction deviate from the cardholder's normal behaviour
    if 'card1' in df.columns and 'TransactionAmt' in df.columns:
        if fit:
            encoders['card1_means'] = ref_df.groupby('card1')['TransactionAmt'].mean()
            encoders['card1_stds']  = ref_df.groupby('card1')['TransactionAmt'].std().fillna(1)
        user_mean = df['card1'].map(encoders['card1_means']).fillna(global_amt_mean)
        user_std  = df['card1'].map(encoders['card1_stds']).fillna(1)
        df['amt_to_mean'] = df['TransactionAmt'] / (user_mean + 1)
        df['amt_dev']     = df['TransactionAmt'] - user_mean
        df['amt_zscore']  = df['amt_dev'] / (user_std + 1e-6)

    # card + billing address combo - a new address for an existing card is suspicious
    if all(x in df.columns for x in ['card1', 'addr1']):
        comb = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
        if fit:
            le = LabelEncoder().fit(comb)
            encoders['card_addr_le']      = le
            encoders['card_addr_classes'] = set(le.classes_)
        le      = encoders['card_addr_le']
        classes = encoders['card_addr_classes']
        mask           = comb.isin(classes)
        result         = np.full(len(comb), -1, dtype=np.int64)
        result[mask]   = le.transform(comb[mask])
        df['card_addr'] = result

    #Velocity Features
    # time since last transaction: fraud often happens in quick bursts
    if 'TransactionDT' in df.columns and 'card1' in df.columns:
        df = df.sort_values('TransactionDT')
        df['time_since_last']     = df.groupby('card1')['TransactionDT'].diff().fillna(0)
        df['log_time_since_last'] = np.log1p(df['time_since_last'])

        if fit:
            encoders['rapid_thresh'] = ref_df.groupby('card1')['TransactionDT'].diff().quantile(0.05)
        rapid_thresh       = encoders.get('rapid_thresh', 60)
        df['is_rapid_txn'] = (df['time_since_last'] < rapid_thresh).astype(np.int8)

    # count of transactions for this card in the last 1h and 24h
    # multiple transactions in a short window is a major fraud red flag
    if 'TransactionDT' in df.columns and 'card1' in df.columns:
        df = df.sort_values('TransactionDT').reset_index(drop=True)

        ONE_HOUR = 3600
        ONE_DAY  = 86400

        txn_dt    = df['TransactionDT'].values
        count_1h  = np.zeros(len(df), dtype=np.int32)
        count_24h = np.zeros(len(df), dtype=np.int32)

        card_groups = df.groupby('card1').indices
        for card, idxs in card_groups.items():
            idxs_sorted = idxs[np.argsort(txn_dt[idxs])]
            times       = txn_dt[idxs_sorted]
            for j, (i, t) in enumerate(zip(idxs_sorted, times)):
                count_1h[i]  = j - np.searchsorted(times, t - ONE_HOUR, 'left')
                count_24h[i] = j - np.searchsorted(times, t - ONE_DAY,  'left')

        df['card1_count_1h']  = count_1h
        df['card1_count_24h'] = count_24h

    # overall transaction count for this card in training history
    if 'card1' in df.columns:
        if fit:
            encoders['card1_velocity'] = ref_df.groupby('card1').size()
        df['card1_txn_velocity'] = df['card1'].map(encoders['card1_velocity']).fillna(1)

    # where does this amount sit relative to the card's historical range
    if 'TransactionAmt' in df.columns and 'card1' in df.columns:
        if fit:
            encoders['card1_max'] = ref_df.groupby('card1')['TransactionAmt'].max()
            encoders['card1_min'] = ref_df.groupby('card1')['TransactionAmt'].min()
        user_max = df['card1'].map(encoders['card1_max']).fillna(df['TransactionAmt'])
        user_min = df['card1'].map(encoders['card1_min']).fillna(df['TransactionAmt'])
        df['amt_range_ratio'] = (df['TransactionAmt'] - user_min) / (user_max - user_min + 1)

    # flag unusually large transactions
    if 'TransactionAmt' in df.columns:
        if fit:
            encoders['amt_95'] = ref_df['TransactionAmt'].quantile(0.95)
            encoders['amt_99'] = ref_df['TransactionAmt'].quantile(0.99)
        df['is_high_amt']    = (df['TransactionAmt'] > encoders['amt_95']).astype(np.int8)
        df['is_extreme_amt'] = (df['TransactionAmt'] > encoders['amt_99']).astype(np.int8)

    #Email Features
    # mismatched purchaser and recipient email domains are a fraud signal
    if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
        df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(np.int8)

    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            if fit:
                encoders[f'{col}_freq'] = ref_df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(encoders[f'{col}_freq']).fillna(0)

    #Card Network / Type Frequency
    # visa/mastercard/debit/credit frequency as a proxy for card type risk
    for col in ['card4', 'card6']:
        if col in df.columns:
            if fit:
                encoders[f'{col}_freq'] = ref_df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(encoders[f'{col}_freq']).fillna(0)

    # Email Domain Fraud Rates
    if 'P_emaildomain' in df.columns:
        if fit:
            encoders['P_emaildomain_count'] = ref_df.groupby('P_emaildomain').size()
            encoders['P_emaildomain_fraud']  = ref_df.groupby('P_emaildomain')['isFraud'].mean()
        df['P_email_txn_count']  = df['P_emaildomain'].map(encoders['P_emaildomain_count']).fillna(0)
        df['P_email_fraud_rate'] = df['P_emaildomain'].map(encoders['P_emaildomain_fraud']).fillna(
            encoders.get('global_fraud_mean', 0.035)
        )

    #Card + Product Combo Fraud Rate
    # certain card/product combinations have higher fraud rates historically
    if 'card1' in df.columns and 'ProductCD' in df.columns:
        comb2 = df['card1'].astype(str) + '_' + df['ProductCD'].astype(str)
        if fit:
            encoders['card_product_fraud'] = (
                ref_df.assign(combo=ref_df['card1'].astype(str) + '_' + ref_df['ProductCD'].astype(str))
                .groupby('combo')['isFraud'].mean()
            )
        df['card_product_fraud_rate'] = comb2.map(encoders['card_product_fraud']).fillna(
            encoders.get('global_fraud_mean', 0.035)
        )

    #Billing Address Fraud Rate
    if 'addr1' in df.columns:
        if fit:
            encoders['addr1_fraud'] = ref_df.groupby('addr1')['isFraud'].mean()
        df['addr1_fraud_rate'] = df['addr1'].map(encoders['addr1_fraud']).fillna(
            encoders.get('global_fraud_mean', 0.035)
        )

    #NEW: C columns (transaction count features)
    # C1-C14 count things like cards associated with an IP, billing address, etc.
    # they're anonymised but are among the strongest fraud signals in this dataset
    c_cols = [c for c in df.columns if c.startswith('C') and c[1:].isdigit()]
    for col in c_cols:
        if fit:
            encoders[f'{col}_mean'] = ref_df[col].mean()
            encoders[f'{col}_std']  = ref_df[col].std()
        col_mean = encoders[f'{col}_mean']
        col_std  = encoders[f'{col}_std']
        # how far is this value from typical?
        df[f'{col}_zscore'] = (df[col] - col_mean) / (col_std + 1e-6)

    # NEW: D columns (time delta features)
    # D1-D15 are days since something (account creation, last login, etc.)
    # very useful for detecting new/throwaway accounts used in fraud
    d_cols = [c for c in df.columns if c.startswith('D') and c[1:].isdigit()]
    for col in d_cols:
        if fit:
            encoders[f'{col}_mean'] = ref_df[col].mean()
            encoders[f'{col}_std']  = ref_df[col].std()
        df[f'{col}_zscore'] = (df[col] - encoders[f'{col}_mean']) / (encoders[f'{col}_std'] + 1e-6)

    # NEW: M columns (match flags)
    # M1-M9 are binary match flags (name on card matches billing name, etc.)
    # count of mismatches is a simple but effective fraud signal
    m_cols = [c for c in df.columns if c.startswith('M') and c[1:].isdigit()]
    if m_cols:
        # T/F encoded as strings, convert to 0/1 first
        m_df = df[m_cols].copy()
        for col in m_cols:
            m_df[col] = m_df[col].map({'T': 1, 'F': 0}).fillna(-1)
        df['m_match_count']    = (m_df == 1).sum(axis=1)
        df['m_mismatch_count'] = (m_df == 0).sum(axis=1)
        df['m_missing_count']  = (m_df == -1).sum(axis=1)

    # NEW: card1 + card2 + addr1 triple combo
    # more granular identity - helps catch cards used at unusual locations
    if all(x in df.columns for x in ['card1', 'card2', 'addr1']):
        triple = (df['card1'].astype(str) + '_'
                  + df['card2'].astype(str) + '_'
                  + df['addr1'].astype(str))
        if fit:
            encoders['triple_fraud'] = (
                ref_df.assign(t=ref_df['card1'].astype(str) + '_'
                              + ref_df['card2'].astype(str) + '_'
                              + ref_df['addr1'].astype(str))
                .groupby('t')['isFraud'].mean()
            )
        df['card1_card2_addr1_fraud_rate'] = triple.map(encoders['triple_fraud']).fillna(
            encoders.get('global_fraud_mean', 0.035)
        )

    # NEW: ProductCD fraud rate
    # W/H/C/S/R products have very different fraud rates
    if 'ProductCD' in df.columns:
        if fit:
            encoders['product_fraud'] = ref_df.groupby('ProductCD')['isFraud'].mean()
        df['product_fraud_rate'] = df['ProductCD'].map(encoders['product_fraud']).fillna(
            encoders.get('global_fraud_mean', 0.035)
        )

    # Device Features
    # smoothed target encoding: prevents overfitting on rare device categories
    # formula from literature: (n * group_mean + k * global_mean) / (n + k)
    for col in ['DeviceType', 'DeviceInfo']:
        if col in df.columns:
            if fit:
                k           = 20
                global_mean = ref_df['isFraud'].mean()
                stats       = ref_df.groupby(col)['isFraud'].agg(['mean', 'count'])
                stats['smooth_te'] = (
                    (stats['mean'] * stats['count'] + global_mean * k)
                    / (stats['count'] + k)
                )
                encoders[f'{col}_te'] = stats['smooth_te']
            df[f'{col}_te'] = df[col].map(encoders[f'{col}_te']).fillna(
                encoders.get('global_fraud_mean', 0.035)
            )

    if fit:
        encoders['global_fraud_mean'] = ref_df['isFraud'].mean()

    return df, encoders


print("\nRunning feature engineering...")
train_data, feature_encoders = engineer_features(train_data, ref_df=train_data, fit=True)
cal_data,   _                = engineer_features(cal_data,   ref_df=train_data, fit=False,
                                                  encoders=feature_encoders)
test_data,  _                = engineer_features(test_data,  ref_df=train_data, fit=False,
                                                  encoders=feature_encoders)



# Step 5: Label Encode Categorical Columns
# XGBoost works with numbers only, so encoding all remaining string columns
# unseen categories in cal/test are assigned -1 (unknown)

cat_cols       = train_data.select_dtypes(include='object').columns.tolist()
label_encoders = {}

for col in cat_cols:
    for df in [train_data, cal_data, test_data]:
        df[col] = df[col].astype(str)
    le = LabelEncoder().fit(train_data[col])
    label_encoders[col] = le
    known_classes   = set(le.classes_)
    train_data[col] = le.transform(train_data[col])
    for df in [cal_data, test_data]:
        mask          = df[col].isin(known_classes)
        encoded       = np.full(len(df), -1, dtype=np.int64)
        encoded[mask] = le.transform(df[col][mask])
        df[col]       = encoded



# Step 6: Drop Near-Zero Variance Features
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
low_var_cols = numeric_cols[train_data[numeric_cols].std() <= 0.01].tolist()
print(f"Dropping {len(low_var_cols)} near-zero variance columns")

for df in [train_data, cal_data, test_data]:
    df.drop(columns=low_var_cols, errors='ignore', inplace=True)

X_train = train_data.drop('isFraud', axis=1)
y_train = train_data['isFraud']
X_cal   = cal_data.drop('isFraud', axis=1)
y_cal   = cal_data['isFraud']
X_test  = test_data.drop('isFraud', axis=1)
y_test  = test_data['isFraud']

# keep only columns that appear in all three sets (some engineered cols might
# not appear if a group had no rows)
common_cols = (X_train.columns
               .intersection(X_cal.columns)
               .intersection(X_test.columns))
X_train, X_cal, X_test = X_train[common_cols], X_cal[common_cols], X_test[common_cols]

# float32 halves memory usage vs float64, no meaningful precision loss here
for df in [X_train, X_cal, X_test]:
    df[:] = df.apply(pd.to_numeric, downcast='float', errors='ignore')

print(f"\nFinal shapes -> Train: {X_train.shape} | Cal: {X_cal.shape} | Test: {X_test.shape}")



# Step 7: Sample Weights
# IEEE fraud dataset has ~3.5% fraud rate => natural ratio is ~27:1
# upweighting fraud by 25x gives the model a proper signal to learn from
# this is the single biggest lever for recall on imbalanced datasets

FRAUD_WEIGHT   = 25.0
sample_weights = np.where(y_train == 1, FRAUD_WEIGHT, 1.0)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
print(f"\nClass ratio (neg/pos): {neg_count/pos_count:.1f}:1")
print(f"Using sample_weight of {FRAUD_WEIGHT}x for fraud transactions")



# Step 8: Train XGBoost
# key hyperparameter choices explained:
#   n_estimators=3000  - more trees helps with recall, early stopping saves us
#   learning_rate=0.02 - slow learning + many trees generally beats fast + few
#   min_child_weight=3 - lower than before so model can learn small fraud clusters
#   gamma=0.1          - lenient split criterion, important for rare fraud patterns
#   reg_alpha/lambda=2 - reduced regularisation compared to original, less conservative
#   subsample=0.8      - row subsampling reduces overfitting
#   colsample_*=0.7    - feature subsampling, makes trees more diverse

model = XGBClassifier(
    objective             = 'binary:logistic',
    n_estimators          = 3000,
    learning_rate         = 0.02,
    max_depth             = 6,
    subsample             = 0.80,
    colsample_bytree      = 0.70,
    colsample_bylevel     = 0.70,
    colsample_bynode      = 0.70,
    reg_alpha             = 2,
    reg_lambda            = 2,
    min_child_weight      = 3,
    gamma                 = 0.1,
    scale_pos_weight      = 1,      # handled via sample_weight instead
    max_delta_step        = 1,
    eval_metric           = 'aucpr',
    tree_method           = 'hist',
    device                = 'cuda', # change to 'cpu' if no GPU available
    early_stopping_rounds = 100,
    random_state          = 42,
    n_jobs                = -1,
)

print("\nTraining XGBoost...")
model.fit(
    X_train, y_train,
    sample_weight = sample_weights,
    eval_set      = [(X_cal, y_cal)],
    verbose       = 100
)

print(f"\nBest iteration: {model.best_iteration}")



# Step 9: Probability Calibration
# XGBoost probabilities on heavily imbalanced data tend to be poorly calibrated
# isotonic regression corrects the probability scale using the held-out cal set

print("\nCalibrating probabilities with isotonic regression...")
calibrated_model = CalibratedClassifierCV(
    model,
    method = 'isotonic',
    cv     = 'prefit'
)
calibrated_model.fit(X_cal, y_cal)



# Step 10: Threshold Selection
# default threshold of 0.5 is useless for imbalanced data
# we sweep on the cal set and pick the threshold that maximises
# F2-score (beta=2 means recall counts 4x more than precision)
# subject to precision staying above 80%
#
# KEY FIX: instead of hard-filtering to p >= 0.80 THEN maximising F2,
# we now compute a penalised score that smoothly degrades below 0.80
# this gives the optimiser room to find better recall without a cliff edge

y_cal_prob = calibrated_model.predict_proba(X_cal)[:, 1]
precision_cal, recall_cal, thresholds_cal = precision_recall_curve(y_cal, y_cal_prob)

TARGET_PRECISION = 0.80
BETA             = 2.0   # F2: recall matters 4x more than precision

p = precision_cal[:-1]
r = recall_cal[:-1]
t = thresholds_cal

# penalised F2: full score above target, drops sharply below
# this avoids the "hard filter kills all good recall candidates" problem
precision_penalty = np.where(p >= TARGET_PRECISION, 1.0, p / TARGET_PRECISION)
fbeta_penalised   = (1 + BETA**2) * (p * r) / (BETA**2 * p + r + 1e-9) * precision_penalty

best              = np.argmax(fbeta_penalised)
best_threshold    = t[best]
best_precision    = p[best]
best_recall       = r[best]

print(f"\nChosen threshold (calibration set):")
print(f"  Threshold : {best_threshold:.4f}")
print(f"  Precision : {best_precision:.4f}")
print(f"  Recall    : {best_recall:.4f}")

# if recall on cal set looks too low, lower the target precision a bit and rerun
if best_recall < 0.55:
    print("\nNote: recall on cal set is below 0.55.")
    print("If test recall is also low, try reducing TARGET_PRECISION to 0.75 or FRAUD_WEIGHT to 20.")


# Step 11: Evaluate on Test Set

y_prob = calibrated_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

print("\n" + "=" * 55)
print("TEST SET RESULTS")
print("=" * 55)
print(f"\nDecision Threshold : {best_threshold:.4f}")
print(classification_report(y_test, y_pred, digits=4,
                             target_names=['Not Fraud', 'Fraud']))

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

tp = int(((y_pred == 1) & (y_test == 1)).sum())
fp = int(((y_pred == 1) & (y_test == 0)).sum())
fn = int(((y_pred == 0) & (y_test == 1)).sum())
print(f"\nTP: {tp}  |  FP: {fp}  |  FN: {fn}")
print(f"Out of every 100 fraud alerts, ~{100*tp/(tp+fp+1e-9):.0f} are actually fraud")
print(f"Catching {100*tp/(tp+fn+1e-9):.0f}% of all fraud cases in the test set")



# Step 12: Threshold Sweep (useful for stakeholders)
# shows the full precision/recall tradeoff so the bank can pick
# a different operating point if their priorities change

print("\nPrecision/Recall tradeoff at different thresholds (test set):")
print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'F2':>8}  {'Alerts':>8}")

precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_test, y_prob)

for thr in np.arange(0.20, 0.85, 0.05):
    idx = np.searchsorted(thresholds_arr, thr)
    if idx >= len(precision_arr) - 1:
        continue
    p_val  = precision_arr[idx]
    r_val  = recall_arr[idx]
    f1_val = 2   * p_val * r_val / (p_val + r_val + 1e-9)
    f2_val = 5   * p_val * r_val / (4 * p_val + r_val + 1e-9)
    alerts = int((y_prob >= thr).sum())
    print(f"{thr:>10.2f}  {p_val:>10.4f}  {r_val:>8.4f}  {f1_val:>8.4f}  {f2_val:>8.4f}  {alerts:>8}")



# Step 13: Plots

cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
axes[0].set_title(f"Confusion Matrix\n(threshold = {best_threshold:.3f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# precision-recall curve with chosen operating point
closest_idx = np.argmin(np.abs(thresholds_arr - best_threshold))
axes[1].plot(recall_arr, precision_arr, lw=2, label='PR Curve')
axes[1].scatter(
    recall_arr[closest_idx], precision_arr[closest_idx],
    color='red', s=120, zorder=5,
    label=f'Chosen  P={precision_arr[closest_idx]:.3f}  R={recall_arr[closest_idx]:.3f}'
)
axes[1].axhline(y=TARGET_PRECISION, color='gray', linestyle='--', alpha=0.6,
                label=f'Target Precision = {TARGET_PRECISION:.0%}')
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve (Test Set)")
axes[1].legend(loc='upper right', fontsize=8)

# precision and recall vs threshold
axes[2].plot(thresholds_arr, precision_arr[:-1], label='Precision', lw=2)
axes[2].plot(thresholds_arr, recall_arr[:-1],    label='Recall',    lw=2)
axes[2].axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'Threshold = {best_threshold:.3f}')
axes[2].set_xlabel("Threshold")
axes[2].set_ylabel("Score")
axes[2].set_title("Precision & Recall vs Threshold")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()



# Step 14: SHAP Interpretability
# SHAP explains WHY the model made each prediction
# important for understanding model behaviour and also for project presentations

print("\nGenerating SHAP values (may take 1-2 minutes)...")
explainer   = shap.TreeExplainer(model)
sample_X    = X_test.sample(min(1000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(sample_X)

# bar chart: which features have the most impact on average
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, sample_X, plot_type='bar', show=False)
plt.tight_layout()
plt.show()

# dot plot: direction and magnitude per feature
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, sample_X, plot_type='dot', show=False)
plt.tight_layout()
plt.show()

# force plot for a single transaction: shows which features pushed toward fraud
sample_idx = np.random.randint(0, len(sample_X))
print(f"\nSHAP force plot for sample index {sample_idx}:")
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx, :],
    sample_X.iloc[sample_idx, :],
    matplotlib=True
)
plt.tight_layout()
plt.show()
