# models/network/train_network_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import shap
import joblib
import os

FEATURES = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
    'sjit', 'djit', 'swin', 'dwin', 'smean', 'dmean',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_ltm',
    'is_sm_ips_ports', 'trans_depth', 'response_body_len'
]

ATTACK_MAP = {
    'Normal': 0,
    'Reconnaissance': 1,
    'DoS': 2,
    'Exploits': 3,
    'Generic': 4,
    'Fuzzers': 5,
    'Backdoors': 6,
    'Shellcode': 7,
    'Worms': 8,
    'Analysis': 9
}

def load_and_prepare(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Encode categorical
    for col in ['proto', 'service', 'state']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        joblib.dump(le, f'models/network/le_{col}.pkl')

    # Map attack category
    df['attack_cat'] = df['attack_cat'].fillna('Normal').str.strip()
    df['attack_label'] = df['attack_cat'].map(ATTACK_MAP).fillna(0).astype(int)

    X = df[FEATURES].fillna(0)
    y_binary = df['label']              # 0=normal, 1=attack
    y_multi = df['attack_label']         # attack category

    return X, y_binary, y_multi

def train(train_path='data/raw/UNSW_NB15_training-set.csv',
          test_path='data/raw/UNSW_NB15_testing-set.csv'):
    
    os.makedirs('models/network', exist_ok=True)
    X, y_binary, y_multi = load_and_prepare(train_path, test_path)

    X_train, X_val, y_train, y_val = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # --- Binary classifier (attack vs normal) ---
    binary_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        class_weight='balanced',   # handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    binary_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    print("Binary AUC:", roc_auc_score(y_val, binary_model.predict_proba(X_val)[:,1]))
    print(classification_report(y_val, binary_model.predict(X_val)))

    # --- Multi-class (attack category) ---
    Xm_train, Xm_val, ym_train, ym_val = train_test_split(X, y_multi, test_size=0.2, random_state=42)
    multi_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=10,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    multi_model.fit(
        Xm_train, ym_train,
        eval_set=[(Xm_val, ym_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Save models
    joblib.dump(binary_model, 'models/network/lgbm_binary.pkl')
    joblib.dump(multi_model, 'models/network/lgbm_multiclass.pkl')
    
    # Save SHAP explainer
    explainer = shap.TreeExplainer(binary_model)
    joblib.dump(explainer, 'models/network/shap_explainer.pkl')
    joblib.dump(FEATURES, 'models/network/features.pkl')
    
    print("✅ Network models saved.")
    return binary_model, multi_model

if __name__ == '__main__':
    train()