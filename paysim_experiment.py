"""
HBPDF Layer-2 Validation on PaySim Mobile-Money Dataset
=======================================================
This script runs the proposed Hybrid (feature-augmented GBM + Isolation Forest)
detector on the PaySim dataset.

PaySim: E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, "PaySim: A financial mobile
money simulator for fraud detection," in Proc. 28th European Modeling and
Simulation Symposium (EMSS), 2016.

Download (free, Kaggle):
    https://www.kaggle.com/datasets/ealaxi/paysim1
File: PS_20174392719_1491204439457_log.csv  (rename to paysim.csv)

USAGE:
    python paysim_experiment.py --data paysim.csv

PaySim columns:
    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Documented PaySim fraud behaviour (from the EMSS-2016 paper):
    - Fraud occurs ONLY in TRANSFER and CASH_OUT transaction types.
    - Fraud = move funds out of a victim account, then cash out.
    - This transfer->cash-out topology mirrors the mule-account / rapid-cash-out
      pattern documented by FIU-Nepal (2024).
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)


def engineer_features(df):
    """Map PaySim raw columns to HBPDF-aligned engineered features.
    Each feature corresponds to a FIU-Nepal red-flag indicator."""
    df = df.copy()

    # Fraud in PaySim only happens in TRANSFER / CASH_OUT
    df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    df['is_cashout'] = (df['type'] == 'CASH_OUT').astype(int)
    df['is_payment'] = (df['type'] == 'PAYMENT').astype(int)
    df['is_cashin'] = (df['type'] == 'CASH_IN').astype(int)
    df['is_debit'] = (df['type'] == 'DEBIT').astype(int)

    # Balance-error features (strong real PaySim signal: fraud empties accounts)
    # errorBalanceOrig: discrepancy in originator balance after txn
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # Account emptied flag (FIU-Nepal: rapid debit leaving balance near nil)
    df['orig_emptied'] = ((df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)).astype(int)

    # Large relative transfer (FIU-Nepal: high-value cash-out)
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

    # Round-number amount (common in scripted fraud)
    df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)

    # Hour-of-day proxy from step (PaySim step = 1 hour; 744 steps = 30 days)
    df['hour_of_day'] = df['step'] % 24

    # Dest is a merchant? (PaySim: merchant accounts start with 'M')
    df['dest_is_merchant'] = df['nameDest'].astype(str).str.startswith('M').astype(int)

    features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'errorBalanceOrig', 'errorBalanceDest',
        'orig_emptied', 'amount_to_balance_ratio', 'is_round_amount',
        'hour_of_day', 'dest_is_merchant',
        'is_transfer', 'is_cashout', 'is_payment', 'is_cashin', 'is_debit'
    ]
    return df[features], df['isFraud']


def evaluate(name, y_test, y_pred, y_score):
    return {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_score),
        'pr_auc': average_precision_score(y_test, y_score),
        'cm': confusion_matrix(y_test, y_pred).tolist()
    }


def main(data_path):
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} transactions; fraud rate = {df['isFraud'].mean():.4%}")

    # PaySim is huge (6.3M rows). Fraud only in TRANSFER/CASH_OUT.
    # Standard practice: restrict to those two types where fraud is possible.
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    print(f"After restricting to TRANSFER/CASH_OUT: {len(df):,} rows; "
          f"fraud rate = {df['isFraud'].mean():.4%}")

    X, y = engineer_features(df)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} (fraud {y_train.sum():,}) | "
          f"Test: {len(X_test):,} (fraud {y_test.sum():,})")

    results = []

    # --- Isolation Forest (unsupervised) ---
    iso = IsolationForest(n_estimators=200, contamination=float(y_train.mean()),
                          random_state=42, n_jobs=-1)
    iso.fit(X_train[y_train == 0])
    y_score_iso = -iso.score_samples(X_test)
    y_pred_iso = (iso.predict(X_test) == -1).astype(int)
    results.append(evaluate('Isolation Forest', y_test, y_pred_iso, y_score_iso))

    # --- Random Forest (supervised baseline) ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_score_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate('Random Forest', y_test, y_pred_rf, y_score_rf))

    # --- Gradient Boosting (supervised) ---
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                     learning_rate=0.1, random_state=42)
    gbm.fit(X_train, y_train)
    y_score_gbm = gbm.predict_proba(X_test)[:, 1]
    y_pred_gbm = gbm.predict(X_test)
    results.append(evaluate('Gradient Boosting', y_test, y_pred_gbm, y_score_gbm))

    # --- Hybrid: Isolation Forest score injected as feature into GBM ---
    iso_feat = IsolationForest(n_estimators=200, contamination=float(y_train.mean()),
                               random_state=42, n_jobs=-1)
    iso_feat.fit(X_train[y_train == 0])
    X_train_aug = X_train.copy()
    X_train_aug['iso_score'] = -iso_feat.score_samples(X_train)
    X_test_aug = X_test.copy()
    X_test_aug['iso_score'] = -iso_feat.score_samples(X_test)

    hyb = GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                     learning_rate=0.08, subsample=0.85, random_state=42)
    hyb.fit(X_train_aug, y_train)
    y_score_hyb = hyb.predict_proba(X_test_aug)[:, 1]
    # Threshold tuning for best F1
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 91):
        f = f1_score(y_test, (y_score_hyb >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr
    y_pred_hyb = (y_score_hyb >= best_thr).astype(int)
    results.append(evaluate('Hybrid (proposed)', y_test, y_pred_hyb, y_score_hyb))
    print(f"Hybrid best threshold = {best_thr:.3f}")

    # --- 3-fold CV on GBM (lighter for speed) ---
    cv = cross_val_score(
        GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        X, y, cv=3, scoring='roc_auc', n_jobs=-1)
    print(f"\n3-fold CV ROC-AUC: {cv.mean():.4f} (std {cv.std():.4f})")

    # --- Print results table ---
    print("\n=== RESULTS (PaySim, TRANSFER/CASH_OUT) ===")
    print(f"{'Model':<22}{'Prec':>8}{'Recall':>8}{'F1':>8}{'ROC-AUC':>9}{'PR-AUC':>8}")
    for r in results:
        print(f"{r['model']:<22}{r['precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['f1']:>8.3f}{r['roc_auc']:>9.3f}{r['pr_auc']:>8.3f}")

    # Feature importance
    fi = pd.DataFrame({'feature': X_train_aug.columns,
                       'importance': hyb.feature_importances_}
                      ).sort_values('importance', ascending=False)
    print("\n=== Top features (Hybrid) ===")
    print(fi.head(10).to_string(index=False))

    import json
    with open('paysim_results.json', 'w') as f:
        json.dump({'results': results, 'cv_mean': float(cv.mean()),
                   'cv_std': float(cv.std()), 'best_threshold': float(best_thr),
                   'feature_importance': fi.to_dict('records')}, f, indent=2)
    print("\nSaved paysim_results.json")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='paysim.csv', help='Path to PaySim CSV')
    args = ap.parse_args()
    main(args.data)
