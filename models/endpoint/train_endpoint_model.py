# models/endpoint/train_endpoint_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import shap
import joblib
import os

FEATURES = [
    'hour_of_day', 'is_weekend', 'process_encoded', 'parent_encoded',
    'bytes_sent_log', 'dst_port', 'is_external_dst', 'username_encoded',
    'event_type_encoded', 'failed_login_rate_1min', 'unique_dst_ips_10min',
    'spawn_depth'
]

def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
    df['bytes_sent_log'] = np.log1p(df['bytes_sent'].fillna(0))
    
    # Is destination external?
    def is_external(ip):
        if pd.isna(ip): return 0
        return 0 if (ip.startswith('10.') or ip.startswith('192.168.') or ip.startswith('172.')) else 1
    df['is_external_dst'] = df['dst_ip'].apply(is_external)

    # Encode categoricals
    for col, out_col in [('process_name', 'process_encoded'), 
                          ('parent_process', 'parent_encoded'),
                          ('username', 'username_encoded'),
                          ('event_type', 'event_type_encoded')]:
        le = LabelEncoder()
        df[out_col] = le.fit_transform(df[col].fillna('unknown').astype(str))
        joblib.dump(le, f'models/endpoint/le_{col}.pkl')

    # Behavioral aggregates (rolling windows)
    df = df.sort_values('timestamp')
    # Simple safe versions (won’t crash)
    df['failed_login_rate_1min'] = (df['event_type'] == 'failed_login').astype(int)

    df['unique_dst_ips_10min'] = df.groupby('username')['dst_ip'].transform('nunique').fillna(1)
    df['spawn_depth'] = df.groupby('parent_pid').cumcount()

    return df

def train(data_path='data/synthetic/endpoint_logs.csv'):
    os.makedirs('models/endpoint', exist_ok=True)
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    X = df[FEATURES].fillna(0)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/endpoint/scaler.pkl')

    # --- Isolation Forest for anomaly score ---
    # Train only on normal traffic
    X_normal = X_scaled[y == 0]
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_normal)
    joblib.dump(iso, 'models/endpoint/isolation_forest.pkl')

    # Anomaly score: -1 = anomaly, 1 = normal → convert to 0-1 probability
    df['anomaly_score'] = iso.decision_function(X_scaled)  # lower = more anomalous
    df['anomaly_prob'] = 1 - (df['anomaly_score'] - df['anomaly_score'].min()) / \
                         (df['anomaly_score'].max() - df['anomaly_score'].min())

    # --- LightGBM on top of anomaly score + features ---
    X_lgbm = pd.DataFrame(X_scaled, columns=FEATURES)
    X_lgbm['anomaly_prob'] = df['anomaly_prob'].values

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_lgbm, y, test_size=0.2, random_state=42)
    
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    
    joblib.dump(lgbm, 'models/endpoint/lgbm_classifier.pkl')
    
    explainer = shap.TreeExplainer(lgbm)
    joblib.dump(explainer, 'models/endpoint/shap_explainer.pkl')
    joblib.dump(list(X_lgbm.columns), 'models/endpoint/features.pkl')

    print("✅ Endpoint models saved.")
    return iso, lgbm

if __name__ == '__main__':
    train()