# AI Threat Detection Engine — Implementation Guide
## Hack Malenadu '26 | Cybersecurity Track

---

## Project Structure

```
threat-engine/
├── data/
│   ├── raw/               # Downloaded public datasets (UNSW-NB15, CICIDS)
│   ├── synthetic/         # Generated endpoint + app logs
│   └── processed/         # Normalized unified event schema CSVs
├── models/
│   ├── network/           # LightGBM trained on UNSW-NB15
│   ├── endpoint/          # Isolation Forest on synthetic endpoint logs
│   ├── application/       # LightGBM on synthetic app logs
│   └── fusion/            # Weighted fusion scorer
├── detection/
│   ├── rules/             # Rule-based detection engine
│   └── ml/                # ML inference wrappers
├── correlation/           # Cross-layer correlation engine
├── scoring/               # Fusion score + severity
├── api/                   # FastAPI real-time ingestion endpoint
└── utils/                 # Normalization, schema helpers
```

---

## STEP 1 — Datasets to Prepare

### Layer 1: Network Layer → UNSW-NB15

**Download from:**
- https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Direct: https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys

**Files you need:**
```
UNSW_NB15_training-set.csv   (~175k rows)
UNSW_NB15_testing-set.csv    (~82k rows)
UNSW-NB15_features.csv       (feature definitions)
```

**What's in it:** 9 attack categories — Fuzzers, Analysis, Backdoors, DoS, Exploits,
Generic, Reconnaissance, Shellcode, Worms + Normal traffic. Label column = `label` (0/1)
and `attack_cat` (string).

**Key features to keep:**
```
dur, proto, service, state, spkts, dpkts, sbytes, dbytes,
rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt,
sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat,
smean, dmean, trans_depth, response_body_len, ct_srv_src,
ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm,
ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd,
ct_src_ltm, ct_srv_dst, is_sm_ips_ports, label, attack_cat
```

---

### Layer 2: Endpoint Layer → Synthetic (generate with script below)

No clean public dataset with labeled malicious endpoint logs exists. The problem statement
explicitly expects synthetic generation. Use `generate_endpoint_logs.py`.

---

### Layer 3: Application Layer → Synthetic (generate with script below)

Use `generate_app_logs.py`. Seed in SQL injection, abnormal user-agents, geo-anomalies,
and an admin bulk-transfer false positive.

---

## STEP 2 — Synthetic Data Generation

### Prompt for LLM-Assisted Generation (copy into Claude or GPT-4)

```
You are a cybersecurity data scientist generating realistic synthetic security logs.

Generate a Python script using the Faker library that creates two CSV files:

=== FILE 1: endpoint_logs.csv ===
Schema: timestamp, hostname, username, process_name, parent_process, parent_pid,
        child_pid, cmdline, file_path, registry_key, event_type, src_ip, dst_ip,
        dst_port, bytes_sent, label, attack_cat

Generate 50,000 rows total with:
- 70% BENIGN: realistic Windows/Linux process activity
  * Common processes: chrome.exe, svchost.exe, python.exe, bash, systemd, cron
  * Admin activity: robocopy, rsync, backup scripts (these are false-positive seeds)
  * Normal file paths: C:\Windows\System32\, /usr/bin/, /home/user/
  * Normal registry: HKLM\SOFTWARE\Microsoft, HKCU\Software

- 10% BRUTE_FORCE: 
  * process_name: sshd, winlogon, lsass
  * event_type: failed_login
  * Burst: >5 failed logins per 60 seconds from same src_ip
  * cmdline patterns: ssh -l root, net use \\target\IPC$

- 10% LATERAL_MOVEMENT:
  * process_name: psexec.exe, wmic.exe, net.exe, mimikatz.exe
  * Unusual parent-child: cmd.exe spawning powershell spawning net.exe
  * dst_ip: internal IPs (10.x.x.x, 192.168.x.x) after initial compromise
  * cmdline: "net use", "psexec \\target", "wmic /node:target"

- 5% DATA_EXFIL:
  * process_name: curl, wget, ftp, rclone, robocopy
  * bytes_sent: > 500MB (flag if admin context = false positive)
  * dst_ip: external IPs
  * file_path: *.zip, *.tar.gz, sensitive paths

- 5% C2_BEACONING:
  * Periodic connections every 30-120 seconds ± 5% jitter
  * bytes_sent: small (100-2000 bytes), consistent
  * dst_ip: same external IP repeated
  * process_name: svchost.exe, explorer.exe (masquerading)

Include 500 rows of FALSE_POSITIVE admin bulk transfer:
- username: admin, sysadmin, backup_user
- process_name: robocopy, rsync, tar
- bytes_sent: 500MB-5GB
- dst_ip: internal backup server
- label: 0 (benign), add column is_fp_candidate: 1

=== FILE 2: app_logs.csv ===
Schema: timestamp, src_ip, user_agent, method, endpoint, status_code, payload_size,
        response_time_ms, geo_country, geo_city, session_id, user_id, 
        auth_header, label, attack_cat

Generate 50,000 rows total with:
- 70% BENIGN: realistic HTTP/API traffic
  * Chrome, Firefox, Safari user agents with versions
  * GET /api/products, POST /api/login, GET /dashboard
  * Status 200, 201, 204
  * payload_size: 200-5000 bytes
  * geo_country: consistent per user_id

- 10% BRUTE_FORCE:
  * endpoint: /api/login, /admin/login, /wp-login.php
  * method: POST
  * status_code: 401, 403 repeated (>10/minute same src_ip)
  * user_agent: Python-requests, Hydra, Medusa, curl
  * payload: {"username": "admin", "password": "...various..."}

- 10% SQL_INJECTION:
  * endpoint: /api/users?id=, /search?q=, /products?filter=
  * payload patterns: ' OR 1=1--, UNION SELECT, DROP TABLE, '; exec(
  * status_code: 500 (error), 200 (successful injection)
  * user_agent: sqlmap/1.x, custom

- 5% ANOMALOUS_GEO:
  * Same user_id login from two countries within 1 hour (impossible travel)
  * geo_country: CN, RU, KP after baseline of IN, US
  * auth_header: valid token (stolen credentials scenario)

- 5% DATA_EXFIL (HTTP):
  * Large response payload: payload_size > 10MB
  * Repeated GET to /api/export, /api/backup, /api/dump
  * Unusual hours: 2AM-5AM

Include 500 rows FALSE_POSITIVE admin export:
- user_id: admin_001, sysadmin
- endpoint: /api/export/report
- payload_size: 10-50MB
- geo_country: same as baseline
- label: 0 (benign), is_fp_candidate: 1

Use realistic timestamps spread over 7 days. 
Randomize but keep attack patterns coherent (same attack_id groups related rows).
Output complete Python code using Faker, pandas, numpy, random.
```

---

## STEP 3 — Model Training Plan

### Model 1: Network Layer — LightGBM (fine-tuned)

**Dataset:** UNSW-NB15  
**Why LightGBM:** Fast, explainable via SHAP, handles class imbalance well, no need to train from scratch.

```python
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
```

---

### Model 2: Endpoint Layer — Isolation Forest + LightGBM

**Dataset:** Your synthetic `endpoint_logs.csv`  
**Why Isolation Forest:** Anomaly detection without needing perfect labels. LightGBM on top for classification.

```python
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
    df['failed_login_rate_1min'] = (
        df.groupby('src_ip')['event_type']
        .transform(lambda x: x.eq('failed_login').rolling('1min', on=df.loc[x.index, 'timestamp']).sum())
        .fillna(0)
    )
    df['unique_dst_ips_10min'] = (
        df.groupby('username')['dst_ip']
        .transform(lambda x: x.rolling(20).apply(lambda w: w.nunique(), raw=False))
        .fillna(1)
    )
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
```

---

### Model 3: Application Layer — LightGBM

**Dataset:** Your synthetic `app_logs.csv`

```python
# models/application/train_app_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shap, joblib, os, re

SQL_INJECTION_PATTERNS = [
    r"'.*OR.*'.*=.*'", r"UNION\s+SELECT", r"DROP\s+TABLE",
    r";\s*exec\s*\(", r"xp_cmdshell", r"--\s*$", r"1=1"
]

def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['is_night'] = df['hour_of_day'].between(0, 5).astype(int)
    df['payload_size_log'] = np.log1p(df['payload_size'].fillna(0))
    df['response_time_log'] = np.log1p(df['response_time_ms'].fillna(0))
    df['is_error_status'] = (df['status_code'] >= 400).astype(int)
    df['is_post'] = (df['method'] == 'POST').astype(int)
    
    # SQLi detection feature
    def has_sqli(payload):
        if pd.isna(payload): return 0
        return int(any(re.search(p, str(payload), re.IGNORECASE) for p in SQL_INJECTION_PATTERNS))
    df['has_sqli_pattern'] = df.get('endpoint', pd.Series([''] * len(df))).apply(has_sqli)

    # Known bad user agents
    bad_agents = ['sqlmap', 'nikto', 'burpsuite', 'python-requests', 'hydra', 'medusa', 'masscan']
    df['is_bad_agent'] = df['user_agent'].fillna('').str.lower().apply(
        lambda ua: int(any(b in ua for b in bad_agents))
    )

    # Impossible travel: same user_id, different geo within 60 min
    df_sorted = df.sort_values(['user_id', 'timestamp'])
    df_sorted['prev_country'] = df_sorted.groupby('user_id')['geo_country'].shift(1)
    df_sorted['prev_time'] = df_sorted.groupby('user_id')['timestamp'].shift(1)
    df_sorted['time_diff_min'] = (df_sorted['timestamp'] - df_sorted['prev_time']).dt.total_seconds() / 60
    df_sorted['impossible_travel'] = (
        (df_sorted['geo_country'] != df_sorted['prev_country']) &
        (df_sorted['time_diff_min'] < 60)
    ).fillna(False).astype(int)
    df = df_sorted

    # Brute force: failed logins per IP per minute
    df['failed_count_1min'] = (
        df[df['status_code'].isin([401, 403])]
        .groupby('src_ip')['timestamp']
        .transform('count')
        .reindex(df.index, fill_value=0)
    )

    # Encode categoricals
    for col, out in [('method', 'method_enc'), ('geo_country', 'country_enc')]:
        le = LabelEncoder()
        df[out] = le.fit_transform(df[col].fillna('unknown').astype(str))
        joblib.dump(le, f'models/application/le_{col}.pkl')

    return df

FEATURES = [
    'hour_of_day', 'is_night', 'payload_size_log', 'response_time_log',
    'is_error_status', 'is_post', 'has_sqli_pattern', 'is_bad_agent',
    'impossible_travel', 'failed_count_1min', 'method_enc', 'country_enc'
]

def train(data_path='data/synthetic/app_logs.csv'):
    os.makedirs('models/application', exist_ok=True)
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    X = df[FEATURES].fillna(0)
    y = df['label']

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=47,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)])

    joblib.dump(model, 'models/application/lgbm_classifier.pkl')
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, 'models/application/shap_explainer.pkl')
    joblib.dump(FEATURES, 'models/application/features.pkl')

    print("✅ Application model saved.")
    return model

if __name__ == '__main__':
    train()
```

---

## STEP 4 — Rule-Based Engine

```python
# detection/rules/rule_engine.py

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class RuleResult:
    fired: bool
    rule_name: str
    attack_cat: str
    confidence: float   # 0 or 1 (rule = deterministic)
    reason: str
    mitre_tag: str

RULES = []

def rule(name, attack_cat, mitre_tag):
    def decorator(fn):
        RULES.append({'name': name, 'attack_cat': attack_cat, 'mitre_tag': mitre_tag, 'fn': fn})
        return fn
    return decorator

# ─── Network Rules ────────────────────────────────────────────────

@rule("brute_force_network", "Brute Force", "T1110")
def brute_force_network(event: dict) -> Optional[RuleResult]:
    """5+ failed auth from same IP within 60s"""
    if event.get('layer') != 'network': return None
    if event.get('failed_login_rate_60s', 0) >= 5:
        return RuleResult(True, "brute_force_network", "Brute Force", 1.0,
            f"Failed logins: {event['failed_login_rate_60s']} in 60s from {event.get('src_ip')}",
            "T1110")
    return None

@rule("c2_beaconing", "C2 Beaconing", "T1071")
def c2_beaconing(event: dict) -> Optional[RuleResult]:
    """Regular low-volume connections, <5% jitter"""
    if event.get('layer') != 'network': return None
    jitter = event.get('beacon_jitter_pct', 100)
    interval = event.get('beacon_interval_s', 0)
    if jitter < 5 and 30 <= interval <= 300 and event.get('sbytes', 9999) < 2000:
        return RuleResult(True, "c2_beaconing", "C2 Beaconing", 1.0,
            f"Beacon interval: {interval}s, jitter: {jitter:.1f}%, bytes: {event.get('sbytes')}",
            "T1071")
    return None

@rule("data_exfil_network", "Data Exfil", "T1041")
def data_exfil_network(event: dict) -> Optional[RuleResult]:
    """Outbound bytes > 3σ above baseline"""
    if event.get('layer') != 'network': return None
    z_score = event.get('outbound_bytes_zscore', 0)
    if z_score > 3 and event.get('is_external_dst', False):
        return RuleResult(True, "data_exfil_network", "Data Exfil", 1.0,
            f"Outbound bytes {z_score:.1f}σ above baseline to external IP",
            "T1041")
    return None

# ─── Endpoint Rules ───────────────────────────────────────────────

@rule("suspicious_spawn", "Lateral Movement", "T1059")
def suspicious_spawn(event: dict) -> Optional[RuleResult]:
    """cmd.exe → powershell → net.exe suspicious chain"""
    if event.get('layer') != 'endpoint': return None
    proc = event.get('process_name', '').lower()
    parent = event.get('parent_process', '').lower()
    suspicious_combos = [
        ('powershell', 'cmd'), ('net', 'powershell'), ('psexec', 'cmd'),
        ('mimikatz', 'powershell'), ('wmic', 'cmd')
    ]
    for child, par in suspicious_combos:
        if child in proc and par in parent:
            return RuleResult(True, "suspicious_spawn", "Lateral Movement", 1.0,
                f"Suspicious process chain: {parent} → {proc}",
                "T1059")
    return None

@rule("admin_bulk_transfer_fp", "False Positive", "N/A")
def admin_bulk_transfer_fp(event: dict) -> Optional[RuleResult]:
    """Known-safe: admin user, internal dst, bulk transfer"""
    if event.get('layer') not in ('endpoint', 'application'): return None
    is_admin = event.get('username', '') in ('admin', 'sysadmin', 'backup_user')
    is_internal = not event.get('is_external_dst', True)
    is_bulk = event.get('bytes_sent', 0) > 100_000_000   # >100MB
    if is_admin and is_internal and is_bulk:
        return RuleResult(True, "admin_bulk_fp", "False Positive", 0.0,
            f"Admin bulk transfer to internal dest — likely legitimate backup",
            "N/A")
    return None

# ─── Application Rules ────────────────────────────────────────────

@rule("sql_injection", "SQLi", "T1190")
def sql_injection(event: dict) -> Optional[RuleResult]:
    import re
    if event.get('layer') != 'application': return None
    payload = str(event.get('endpoint', '') + event.get('user_agent', ''))
    patterns = [r"'.*OR.*'.*=.*'", r"UNION\s+SELECT", r"DROP\s+TABLE", r";\s*exec"]
    for p in patterns:
        if re.search(p, payload, re.IGNORECASE):
            return RuleResult(True, "sql_injection", "SQLi", 1.0,
                f"SQL injection pattern detected in request",
                "T1190")
    return None

@rule("impossible_travel", "Account Takeover", "T1078")
def impossible_travel_rule(event: dict) -> Optional[RuleResult]:
    if event.get('layer') != 'application': return None
    if event.get('impossible_travel', False):
        return RuleResult(True, "impossible_travel", "Account Takeover", 1.0,
            f"User {event.get('user_id')} login from {event.get('geo_country')} after {event.get('prev_country')} < 60min",
            "T1078")
    return None

def evaluate_rules(event: dict) -> list[RuleResult]:
    results = []
    for rule_def in RULES:
        result = rule_def['fn'](event)
        if result and result.fired:
            results.append(result)
    return results
```

---

## STEP 5 — Fusion Scoring Engine

```python
# scoring/fusion.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class FusionScore:
    score: float              # 0.0 – 1.0
    severity: str             # Low / Medium / High / Critical
    rule_fired: bool
    ml_probability: float
    layer_count: int          # How many layers corroborated
    is_fp_candidate: bool
    mitre_tag: str
    attack_cat: str
    explanation_features: list[tuple[str, float]]  # SHAP features

SEVERITY_THRESHOLDS = {
    'Low':      (0.0, 0.35),
    'Medium':   (0.35, 0.60),
    'High':     (0.60, 0.85),
    'Critical': (0.85, 1.01)
}

def compute_fusion_score(
    rule_results: list,
    ml_probability: float,
    layer_results: dict,       # {'network': 0.7, 'endpoint': 0.8, 'application': None}
    event: dict,
    shap_values: list[tuple]   # [(feature_name, shap_value)]
) -> FusionScore:
    
    rule_hit = 1.0 if rule_results else 0.0
    
    # Base score formula
    base_score = 0.4 * rule_hit + 0.6 * ml_probability
    
    # Cross-layer boost: +0.2 per additional corroborated layer
    active_layers = [l for l, v in layer_results.items() if v is not None and v > 0.4]
    layer_boost = max(0, len(active_layers) - 1) * 0.2
    
    score = min(1.0, base_score + layer_boost)
    
    # False positive suppression
    is_fp = False
    fp_rules = [r for r in rule_results if r.attack_cat == "False Positive"]
    if fp_rules:
        is_fp = True
        score = min(score, 0.59)   # cap at Medium regardless
    
    # Also cap if matches known-safe profile
    if _matches_safe_profile(event) and not rule_results:
        is_fp = True
        score = min(score, 0.59)
    
    # Determine severity
    severity = 'Low'
    for sev, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= score < hi:
            severity = sev
            break
    
    # Dominant rule
    mitre = rule_results[0].mitre_tag if rule_results else "N/A"
    attack_cat = rule_results[0].attack_cat if rule_results else _infer_attack_cat(ml_probability, event)
    
    # Top SHAP features (magnitude sort)
    top_features = sorted(shap_values, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    return FusionScore(
        score=round(score, 3),
        severity=severity,
        rule_fired=bool(rule_results),
        ml_probability=round(ml_probability, 3),
        layer_count=len(active_layers),
        is_fp_candidate=is_fp,
        mitre_tag=mitre,
        attack_cat=attack_cat,
        explanation_features=top_features
    )

def _matches_safe_profile(event: dict) -> bool:
    return (
        event.get('username') in ('admin', 'sysadmin', 'backup_user') and
        not event.get('is_external_dst', True) and
        event.get('src_ip', '').startswith(('10.', '192.168.'))
    )

def _infer_attack_cat(prob: float, event: dict) -> str:
    if prob < 0.35: return "Normal"
    layer = event.get('layer', 'unknown')
    return f"Anomaly ({layer})"

def severity_from_score(score: float) -> str:
    for sev, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= score < hi:
            return sev
    return 'Low'
```

---

## STEP 6 — SHAP Explanation → Plain English

```python
# scoring/explainer.py
#
# LLM backend: Ollama (free, runs locally)
# Install Ollama:  https://ollama.com/download
# Pull a model:    ollama pull mistral        (4GB, good quality)
#                  ollama pull llama3.1:8b    (5GB, better quality)
#                  ollama pull phi3:mini      (2GB, fastest, lower quality)
# Start server:    ollama serve               (runs on http://localhost:11434)

import shap
import numpy as np
import joblib
import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"   # change to "llama3.1:8b" or "phi3:mini" as needed

FEATURE_EXPLANATIONS = {
    'sbytes':                 'outbound bytes sent',
    'dbytes':                 'inbound bytes received',
    'rate':                   'packet rate (packets/sec)',
    'sjit':                   'source jitter (timing irregularity)',
    'djit':                   'destination jitter',
    'ct_srv_src':             'connection count to same service/source',
    'failed_login_rate_1min': 'failed login attempts per minute',
    'has_sqli_pattern':       'SQL injection pattern in request',
    'is_bad_agent':           'suspicious user agent detected',
    'impossible_travel':      'geographically impossible login',
    'bytes_sent_log':         'bytes sent (log scale)',
    'is_external_dst':        'traffic going to external IP',
    'anomaly_prob':           'anomaly isolation score',
    'payload_size_log':       'request payload size (log scale)',
}

def _check_ollama() -> bool:
    """Returns True if Ollama server is reachable"""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def get_shap_explanation(model_name: str, features: dict) -> list[tuple]:
    """Returns top-5 (feature, shap_value) tuples"""
    try:
        explainer = joblib.load(f'models/{model_name}/shap_explainer.pkl')
        feature_list = joblib.load(f'models/{model_name}/features.pkl')
        X = np.array([[features.get(f, 0) for f in feature_list]])
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1 for binary
        return list(zip(feature_list, shap_vals[0]))
    except Exception as e:
        print(f"SHAP error: {e}")
        return []

def shap_to_english(shap_features: list[tuple]) -> str:
    """Convert SHAP values to plain English bullets"""
    lines = []
    for feat, val in sorted(shap_features, key=lambda x: abs(x[1]), reverse=True)[:5]:
        direction = "increased" if val > 0 else "decreased"
        readable = FEATURE_EXPLANATIONS.get(feat, feat.replace('_', ' '))
        lines.append(f"• {readable} {direction} the threat score by {abs(val):.3f}")
    return "\n".join(lines)

def generate_playbook(incident_context: dict, shap_explanation: str) -> str:
    """
    Generate incident response playbook using local Ollama LLM.
    Falls back to a rule-based static playbook if Ollama is not running.
    """
    if not _check_ollama():
        return _static_playbook(incident_context, shap_explanation)

    prompt = f"""You are a senior SOC analyst. An incident has been detected:

Attack Category: {incident_context.get('attack_cat', 'Unknown')}
Severity: {incident_context.get('severity', 'Unknown')}
MITRE ATT&CK: {incident_context.get('mitre_tag', 'N/A')}
Fusion Score: {incident_context.get('score', 0):.2f}
Layers Corroborated: {incident_context.get('layer_count', 1)}
Source IP: {incident_context.get('src_ip', 'N/A')}
Is False Positive Candidate: {incident_context.get('is_fp_candidate', False)}

Why it was flagged (top SHAP features):
{shap_explanation}

Generate a concise, actionable incident response playbook with these 5 sections:
1. Immediate Actions (first 15 minutes)
2. Investigation Steps (what logs to pull, what to check)
3. Containment (block, isolate, or flag)
4. False Positive Check (how to verify if this is benign)
5. Prevention (long-term fix)

Use real command examples. Be specific. Keep it under 400 words. Do not use markdown headers."""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 600}
            },
            timeout=60
        )
        result = response.json()
        return result.get("response", _static_playbook(incident_context, shap_explanation))
    except Exception as e:
        print(f"Ollama error: {e}")
        return _static_playbook(incident_context, shap_explanation)

def _static_playbook(incident_context: dict, shap_explanation: str) -> str:
    """
    Rule-based fallback playbook — no LLM needed.
    Used when Ollama is not running or times out.
    """
    attack = incident_context.get('attack_cat', 'Unknown')
    src_ip = incident_context.get('src_ip', 'N/A')
    severity = incident_context.get('severity', 'Unknown')
    mitre = incident_context.get('mitre_tag', 'N/A')
    is_fp = incident_context.get('is_fp_candidate', False)

    PLAYBOOKS = {
        'Brute Force': {
            'immediate':    f"Block {src_ip} at firewall immediately.\n  iptables -A INPUT -s {src_ip} -j DROP",
            'investigate':  "Pull auth logs: journalctl -u sshd | grep 'Failed'\nCheck /var/log/auth.log for pattern",
            'contain':      f"Add {src_ip} to blocklist. Lock targeted accounts temporarily.\n  usermod -L <username>",
            'fp_check':     "Verify if src_ip belongs to a known scanner or internal security team test.",
            'prevent':      "Enable fail2ban. Enforce MFA. Move SSH to non-standard port."
        },
        'C2 Beaconing': {
            'immediate':    f"Isolate the beaconing host from network. Kill the suspicious process.",
            'investigate':  "Run: ss -tnp | grep <suspicious_pid>\nCheck DNS history: cat /var/log/syslog | grep <c2_ip>",
            'contain':      f"Block outbound to {src_ip} on all ports at perimeter firewall.",
            'fp_check':     "Verify if process is a legitimate monitoring agent (Datadog, Zabbix, etc.)",
            'prevent':      "Deploy EDR. Block known C2 IP ranges. Enable DNS-over-HTTPS monitoring."
        },
        'Data Exfil': {
            'immediate':    "Terminate the exfiltration process immediately. Revoke network access for host.",
            'investigate':  "Check: lsof -p <pid> to see open files.\naudit log: ausearch -k file_access",
            'contain':      f"Block outbound to {src_ip}. Preserve disk image for forensics.",
            'fp_check':     "Check if user is admin doing a scheduled backup to an authorized destination.",
            'prevent':      "Implement DLP (Data Loss Prevention). Classify sensitive files. Restrict large uploads."
        },
        'Lateral Movement': {
            'immediate':    "Isolate the compromised host from the internal network immediately.",
            'investigate':  "Check: Get-WinEvent -LogName Security | Where {$_.Id -eq 4688}\nLook for psexec, wmic, net use events",
            'contain':      "Disable compromised credentials. Segment affected subnet.",
            'fp_check':     "Confirm with sysadmin if they ran remote admin commands at this time.",
            'prevent':      "Enforce least privilege. Disable admin shares. Deploy PAM solution."
        },
        'SQLi': {
            'immediate':    "Enable WAF block mode immediately if in detection mode.",
            'investigate':  "Grep web server logs: grep -E \"('|UNION|SELECT|DROP|exec)\" /var/log/nginx/access.log",
            'contain':      f"Block {src_ip} at WAF/firewall. Check if any rows were modified.",
            'fp_check':     "Check if this is a scheduled pen-test or vulnerability scan by security team.",
            'prevent':      "Parameterize all DB queries. Deploy ModSecurity WAF. Enable query logging."
        },
        'Account Takeover': {
            'immediate':    "Force logout all sessions for the affected user_id. Reset credentials.",
            'investigate':  "Pull login history for the user. Check source IPs vs usual geo-baseline.",
            'contain':      "Revoke active tokens. Enable MFA requirement for the account.",
            'fp_check':     "Check if user is traveling or using a VPN that exits in a different country.",
            'prevent':      "Implement impossible-travel detection alerts. Use risk-based authentication."
        },
    }

    playbook = PLAYBOOKS.get(attack, {
        'immediate':   'Investigate the flagged event manually.',
        'investigate': 'Pull relevant logs for the affected host and time window.',
        'contain':     'Isolate the source if threat is confirmed.',
        'fp_check':    'Verify context with the asset owner.',
        'prevent':     'Review and tighten relevant security controls.'
    })

    fp_note = "\n  ⚠️  FALSE POSITIVE CANDIDATE: This event matches a known-safe profile. Verify before acting." if is_fp else ""

    return f"""INCIDENT RESPONSE PLAYBOOK
Attack: {attack} | Severity: {severity} | MITRE: {mitre}{fp_note}

Why flagged:
{shap_explanation}

1. IMMEDIATE ACTIONS
  {playbook['immediate']}

2. INVESTIGATION STEPS
  {playbook['investigate']}

3. CONTAINMENT
  {playbook['contain']}

4. FALSE POSITIVE CHECK
  {playbook['fp_check']}

5. PREVENTION
  {playbook['prevent']}
"""
```

---

## STEP 7 — Real-Time Ingestion API

```python
# api/main.py
# Run: uvicorn api.main:app --reload --port 8000

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Literal
import asyncio
import joblib
import numpy as np
from detection.rules.rule_engine import evaluate_rules
from scoring.fusion import compute_fusion_score
from scoring.explainer import get_shap_explanation, shap_to_english, generate_playbook
import time

app = FastAPI(title="Threat Detection Engine")

# Load models once at startup
@app.on_event("startup")
async def load_models():
    app.state.models = {
        'network': {
            'binary': joblib.load('models/network/lgbm_binary.pkl'),
            'features': joblib.load('models/network/features.pkl'),
        },
        'endpoint': {
            'classifier': joblib.load('models/endpoint/lgbm_classifier.pkl'),
            'iso_forest': joblib.load('models/endpoint/isolation_forest.pkl'),
            'scaler': joblib.load('models/endpoint/scaler.pkl'),
            'features': joblib.load('models/endpoint/features.pkl'),
        },
        'application': {
            'classifier': joblib.load('models/application/lgbm_classifier.pkl'),
            'features': joblib.load('models/application/features.pkl'),
        }
    }
    print("✅ All models loaded.")

class NetworkEvent(BaseModel):
    layer: Literal['network'] = 'network'
    src_ip: str
    dst_ip: str
    proto: str
    sbytes: float
    dbytes: float
    rate: float
    dur: float
    failed_login_rate_60s: Optional[float] = 0
    beacon_jitter_pct: Optional[float] = 100
    beacon_interval_s: Optional[float] = 0
    outbound_bytes_zscore: Optional[float] = 0
    is_external_dst: Optional[bool] = False
    # ... other network features

class EndpointEvent(BaseModel):
    layer: Literal['endpoint'] = 'endpoint'
    hostname: str
    username: str
    process_name: str
    parent_process: str
    bytes_sent: Optional[float] = 0
    dst_ip: Optional[str] = None
    dst_port: Optional[int] = 0
    event_type: str

class AppEvent(BaseModel):
    layer: Literal['application'] = 'application'
    src_ip: str
    user_agent: str
    method: str
    endpoint: str
    status_code: int
    payload_size: float
    geo_country: str
    user_id: Optional[str] = None

@app.post("/ingest/network")
async def ingest_network(event: NetworkEvent):
    return await _process_event(event.dict(), 'network')

@app.post("/ingest/endpoint")
async def ingest_endpoint(event: EndpointEvent):
    return await _process_event(event.dict(), 'endpoint')

@app.post("/ingest/application")
async def ingest_app(event: AppEvent):
    return await _process_event(event.dict(), 'application')

async def _process_event(event: dict, layer: str):
    start = time.time()
    
    # 1. Rule-based evaluation
    rule_results = evaluate_rules(event)
    
    # 2. ML inference
    ml_prob = _ml_predict(event, layer)
    
    # 3. SHAP explanation
    shap_vals = get_shap_explanation(layer, event)
    shap_text = shap_to_english(shap_vals)
    
    # 4. Fusion score
    layer_scores = {layer: ml_prob, **{l: None for l in ('network', 'endpoint', 'application') if l != layer}}
    fusion = compute_fusion_score(rule_results, ml_prob, layer_scores, event, shap_vals)
    
    # 5. Generate playbook if High/Critical
    playbook = None
    if fusion.severity in ('High', 'Critical'):
        playbook = generate_playbook({
            'attack_cat': fusion.attack_cat,
            'severity': fusion.severity,
            'mitre_tag': fusion.mitre_tag,
            'score': fusion.score,
            'layer_count': fusion.layer_count,
            'src_ip': event.get('src_ip'),
            'is_fp_candidate': fusion.is_fp_candidate
        }, shap_text)
    
    return {
        "incident": {
            "score": fusion.score,
            "severity": fusion.severity,
            "attack_cat": fusion.attack_cat,
            "mitre_tag": fusion.mitre_tag,
            "rule_fired": fusion.rule_fired,
            "ml_probability": fusion.ml_probability,
            "layer_count": fusion.layer_count,
            "is_fp_candidate": fusion.is_fp_candidate,
        },
        "explanation": {
            "shap_text": shap_text,
            "top_features": [(f, round(v, 4)) for f, v in fusion.explanation_features],
            "rule_reasons": [r.reason for r in rule_results],
        },
        "playbook": playbook,
        "processing_ms": round((time.time() - start) * 1000, 2)
    }

def _ml_predict(event: dict, layer: str) -> float:
    """Extract features and run ML inference for the given layer"""
    try:
        models = app.state.models[layer]
        features = models['features']
        X = np.array([[event.get(f, 0) for f in features]])
        
        if layer == 'endpoint':
            X_scaled = models['scaler'].transform(X)
            prob = models['classifier'].predict_proba(X_scaled)[0][1]
        else:
            prob = models[list(models.keys())[0]].predict_proba(X)[0][1]
        return float(prob)
    except Exception as e:
        print(f"ML inference error: {e}")
        return 0.0

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": hasattr(app.state, 'models')}
```

---

## STEP 8 — Setup & Run

```bash
# requirements.txt  (zero paid APIs — fully open source)
lightgbm==4.3.0
scikit-learn==1.4.0
shap==0.45.0
pandas==2.2.0
numpy==1.26.0
fastapi==0.110.0
uvicorn==0.27.0
requests==2.31.0    # for Ollama HTTP calls
faker==24.0.0
joblib==1.3.2
pyarrow==15.0.0

# Install Python deps
pip install -r requirements.txt

# ── Install Ollama (free local LLM — one-time setup) ──────────────
# Linux/macOS:
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose one based on your RAM):
ollama pull mistral        # 4GB RAM — recommended, good quality
ollama pull llama3.1:8b   # 5GB RAM — better quality
ollama pull phi3:mini      # 2GB RAM — fastest, lower quality

# Start Ollama server (keep this running in a separate terminal):
ollama serve

# ── Download UNSW-NB15 (manual step) ──────────────────────────────
# Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
# Download and place:
#   data/raw/UNSW_NB15_training-set.csv
#   data/raw/UNSW_NB15_testing-set.csv

# ── Generate synthetic data ────────────────────────────────────────
python utils/generate_endpoint_logs.py   # outputs data/synthetic/endpoint_logs.csv
python utils/generate_app_logs.py        # outputs data/synthetic/app_logs.csv

# ── Train all models (run in order) ───────────────────────────────
python models/network/train_network_model.py
python models/endpoint/train_endpoint_model.py
python models/application/train_app_model.py

# ── Evaluate models (see STEP 9) ──────────────────────────────────
python evaluation/evaluate_all.py

# ── Start API ─────────────────────────────────────────────────────
uvicorn api.main:app --reload --port 8000

# ── Test ──────────────────────────────────────────────────────────
curl -X POST http://localhost:8000/ingest/network \
  -H "Content-Type: application/json" \
  -d '{"src_ip":"192.168.1.100","dst_ip":"8.8.8.8","proto":"tcp","sbytes":50000,"dbytes":1000,"rate":200,"dur":0.5,"failed_login_rate_60s":8}'
```

---

## Dataset Summary Table

| Layer | Dataset | Source | Size | Notes |
|-------|---------|--------|------|-------|
| Network | UNSW-NB15 | UNSW Research | ~257k rows | 9 attack categories, best for this use case |
| Network (alt) | CICIDS-2017 | UNB | ~2M rows | DDoS, brute force, infiltration |
| Endpoint | Synthetic (generate_endpoint_logs.py) | Generated | 50k rows | Process chains, lateral movement |
| Application | Synthetic (generate_app_logs.py) | Generated | 50k rows | SQLi, brute force, impossible travel |

---

## Model Summary Table

| Model | File | Trained On | Detects | Output |
|-------|------|-----------|---------|--------|
| LightGBM Binary | `models/network/lgbm_binary.pkl` | UNSW-NB15 | Attack vs Normal | 0-1 probability |
| LightGBM Multiclass | `models/network/lgbm_multiclass.pkl` | UNSW-NB15 | 9 attack categories | Category + confidence |
| Isolation Forest | `models/endpoint/isolation_forest.pkl` | Synthetic benign | Novel anomalies | Anomaly score |
| LightGBM Classifier | `models/endpoint/lgbm_classifier.pkl` | Synthetic endpoint | 4 attack categories | 0-1 probability |
| LightGBM Classifier | `models/application/lgbm_classifier.pkl` | Synthetic app | SQLi, brute force, anomalous geo | 0-1 probability |
| SHAP Explainer | Each model folder | Wraps above models | Any | Feature attribution |

---

## Docker (for later)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install Ollama inside container
RUN curl -fsSL https://ollama.com/install.sh | sh
COPY . .
# Models must be pre-trained and copied in, or mounted as volume
EXPOSE 8000
CMD ["sh", "-c", "ollama serve & sleep 3 && ollama pull mistral && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  threat-engine:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models    # mount pre-trained models
      - ./data:/app/data
      - ollama_data:/root/.ollama  # persist downloaded LLM weights

volumes:
  ollama_data:
```

> **Tip:** For docker, it's faster to run Ollama as a separate service container
> rather than inside the app container — it can share the model across restarts.
