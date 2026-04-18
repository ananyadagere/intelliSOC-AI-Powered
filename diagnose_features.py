"""
Run this in your project root:
  python diagnose_features.py

It prints exactly what feature names your models expect vs what
your event dict provides, so you can fix the mapping.
"""
import joblib
import os

LAYERS = {
    'network':     'models/network/features.pkl',
    'endpoint':    'models/endpoint/features.pkl',
    'application': 'models/application/features.pkl',
}

# Typical event keys you send from the frontend
EVENT_KEYS = {
    'network': [
        'src_ip','dst_ip','proto','sbytes','dbytes','rate','dur',
        'failed_login_rate_60s','beacon_jitter_pct','beacon_interval_s',
        'outbound_bytes_zscore','is_external_dst'
    ],
    'endpoint': [
        'hostname','username','process_name','parent_process',
        'bytes_sent','dst_ip','dst_port','event_type'
    ],
    'application': [
        'src_ip','user_agent','method','endpoint','status_code',
        'payload_size','geo_country','user_id'
    ]
}

for layer, path in LAYERS.items():
    if not os.path.exists(path):
        print(f"\n[{layer.upper()}] ❌ File not found: {path}")
        continue
    features = joblib.load(path)
    event_keys = set(EVENT_KEYS[layer])
    feat_set = set(features)
    print(f"\n{'='*60}")
    print(f"[{layer.upper()}]  {len(features)} features expected")
    print(f"  Features list: {features}")
    missing = feat_set - event_keys
    extra   = event_keys - feat_set
    print(f"  ✅ Matched: {feat_set & event_keys}")
    print(f"  ❌ In model but NOT in event: {missing}")
    print(f"  ⚠️  In event but NOT in model: {extra}")

# Also check endpoint scaler
scaler_path = 'models/endpoint/scaler.pkl'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f"\n[ENDPOINT SCALER]  n_features_in_={scaler.n_features_in_}")
    ep_features = joblib.load('models/endpoint/features.pkl')
    print(f"[ENDPOINT FEATURES] count={len(ep_features)}")
    if scaler.n_features_in_ != len(ep_features):
        print(f"  ❌ MISMATCH: scaler expects {scaler.n_features_in_}, features.pkl has {len(ep_features)}")
        print("  → Retrain scaler with same feature list, or trim features.pkl to match.")