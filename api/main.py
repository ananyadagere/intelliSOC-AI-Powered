# Run: uvicorn api.main:app --reload --port 8000

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Literal
import joblib
import numpy as np
import re
import ipaddress
import time
from datetime import datetime
from contextlib import asynccontextmanager

import detection.rules.rule_engine as rule_engine
import scoring.fusion as fusion
import scoring.explainer as explainer

from fastapi.middleware.cors import CORSMiddleware

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match training scripts exactly
# ─────────────────────────────────────────────────────────────────────────────

PROTO_MAP = {'tcp': 1, 'udp': 2, 'icmp': 3}

EVENT_TYPE_MAP = {
    'process_start': 0, 'failed_login': 1, 'file_transfer': 2,
    'network_connect': 3, 'registry_write': 4
}

KNOWN_ATTACK_AGENTS = {
    'sqlmap', 'nikto', 'nmap', 'masscan', 'hydra', 'metasploit',
    'burpsuite', 'dirbuster', 'zgrab', 'medusa', 'gobuster',
    'wfuzz', 'python-requests', 'curl/', 'go-http-client', 'axios', 'wget',
}

AMBIGUOUS_AGENTS = {'python-requests', 'curl/', 'go-http-client', 'axios', 'wget'}

SUSPICIOUS_PROCS = {
    'powershell.exe', 'cmd.exe', 'wscript.exe',
    'mshta.exe', 'cscript.exe', 'regsvr32.exe',
}

UNUSUAL_PARENTS = {
    'winword.exe', 'excel.exe', 'outlook.exe',
    'iexplore.exe', 'chrome.exe', 'firefox.exe',
}

FOREIGN_SET = {'CN', 'RU', 'KP', 'IR', 'NG', 'UA'}

SERVICE_MAP = {'http': 1, 'https': 2, 'ftp': 3, 'smtp': 4, 'dns': 5, 'ssh': 6}
STATE_MAP   = {'FIN': 1, 'INT': 2, 'CON': 3, 'REQ': 4, 'RST': 5, 'URN': 6}

SQL_INJECTION_PATTERNS = re.compile(
    r"('[^']*OR[^']'|UNION\s+SELECT|DROP\s+TABLE|;\s*exec\s*\(|xp_cmdshell"
    r"|--\s*$|1=1|ORDER\s+BY\s+\d+--|%27|information_schema)",
    re.IGNORECASE
)


def _is_internal_ip(ip_str: str) -> int:
    try:
        return int(ipaddress.ip_address(ip_str).is_private)
    except Exception:
        return 0


def _country_encode(country: str, le) -> int:
    """Use the saved LabelEncoder; fall back to len(classes_) for unknowns."""
    try:
        return int(le.transform([country])[0])
    except ValueError:
        return len(le.classes_)


def _method_encode(method: str, le) -> int:
    try:
        return int(le.transform([method])[0])
    except ValueError:
        return 0


def encode_event(event: dict, features: list, aux: dict) -> np.ndarray:
    """
    Build the feature vector the model expects.
    `aux` carries loaded encoders: aux['le_country'], aux['le_method']
    """
    ev = dict(event)
    now = datetime.utcnow()

    # ── Raw numerics ───────────────────────────────────────────────────────
    sbytes      = float(ev.get('sbytes', 0) or 0)
    dbytes      = float(ev.get('dbytes', 0) or 0)
    rate        = float(ev.get('rate', 0) or 0)
    dur         = float(ev.get('dur', 0) or 0)
    payload_sz  = float(ev.get('payload_size', 0) or 0)
    status_code = int(ev.get('status_code', 200) or 200)
    bytes_sent  = float(ev.get('bytes_sent', sbytes) or 0)
    dst_port    = int(ev.get('dst_port', 0) or 0)

    # ── Temporal (match training exactly) ─────────────────────────────────
    hour_of_day = now.hour
    is_weekend  = int(now.weekday() >= 5)
    is_night    = int(hour_of_day < 6 or hour_of_day >= 23)

    # ── Log-transforms ─────────────────────────────────────────────────────
    bytes_sent_log = float(np.log1p(bytes_sent))
    payload_log    = float(np.log1p(payload_sz))
    resp_time_log  = float(np.log1p(float(ev.get('resp_time_ms', 0) or 0)))
    src_ip_req_log = float(np.log1p(float(ev.get('src_ip_req_count', 1) or 1)))

    # ── Network protocol ───────────────────────────────────────────────────
    proto_enc   = PROTO_MAP.get(str(ev.get('proto', '')).lower(), 0)
    service_enc = SERVICE_MAP.get(str(ev.get('service', '')).lower(), 0)
    state_enc   = STATE_MAP.get(str(ev.get('state', '')).upper(), 0)

    # ── HTTP method ────────────────────────────────────────────────────────
    method_raw  = str(ev.get('method', 'GET')).upper()
    method_enc  = _method_encode(method_raw, aux['le_method'])
    is_post     = int(method_raw == 'POST')
    is_get      = int(method_raw == 'GET')

    # ── HTTP status ────────────────────────────────────────────────────────
    is_error      = int(status_code >= 400)
    is_server_err = int(status_code >= 500)
    is_large_resp = int(payload_sz > 1_000_000)

    # ── User-agent ─────────────────────────────────────────────────────────
    ua = str(ev.get('user_agent', '')).lower()
    has_known_attack_agent = int(any(a in ua for a in KNOWN_ATTACK_AGENTS))
    has_ambiguous_agent    = int(any(a in ua for a in AMBIGUOUS_AGENTS))
    has_browser_agent      = int(any(b in ua for b in ('mozilla', 'webkit', 'gecko', 'safari', 'chrome')))

    # ── Endpoint / URL features ────────────────────────────────────────────
    endpoint_str          = str(ev.get('endpoint', ''))
    has_sqli_pattern      = int(bool(SQL_INJECTION_PATTERNS.search(endpoint_str)))
    endpoint_has_query    = int('?' in endpoint_str)
    endpoint_query_length = len(endpoint_str.split('?')[1]) if '?' in endpoint_str else 0
    path_base             = endpoint_str.split('?')[0].lower()
    is_admin_endpoint     = int(bool(re.search(r'/admin|/backup|/export|/dump', path_base)))
    is_auth_endpoint      = int(bool(re.search(r'/login|/auth|/signin', path_base)))

    # ── IP / geo ───────────────────────────────────────────────────────────
    src_ip_str  = str(ev.get('src_ip', ''))
    dst_ip_str  = str(ev.get('dst_ip', ''))
    is_internal_ip = _is_internal_ip(src_ip_str)
    is_external_dst = (
        int(not _is_internal_ip(dst_ip_str))
        if dst_ip_str
        else int(bool(ev.get('is_external_dst', False)))
    )

    country_raw        = str(ev.get('geo_country', 'US')).upper()
    country_enc        = _country_encode(country_raw, aux['le_country'])
    is_foreign_country = int(country_raw in FOREIGN_SET)

    # ── Process / endpoint (endpoint layer) ───────────────────────────────
    pname           = str(ev.get('process_name', '')).lower()
    process_encoded = int(pname in SUSPICIOUS_PROCS)

    parent         = str(ev.get('parent_process', '')).lower()
    parent_encoded = int(parent in UNUSUAL_PARENTS)

    event_type_encoded = EVENT_TYPE_MAP.get(str(ev.get('event_type', '')).lower(), 0)

    username         = str(ev.get('username', ''))
    username_encoded = abs(hash(username)) % 1000 if username else 0

    # ── Behavioural counters ───────────────────────────────────────────────
    failed_login_rate_1min = float(
        ev.get('failed_login_rate_60s', ev.get('failed_login_rate_1min', 0)) or 0
    )
    unique_dst_ips_10min = float(ev.get('unique_dst_ips_10min', 0) or 0)
    spawn_depth          = float(ev.get('spawn_depth', 0) or 0)
    fail_rate_cumulative = float(ev.get('fail_rate_cumulative', failed_login_rate_1min) or 0)
    anomaly_prob         = float(ev.get('anomaly_prob', 0) or 0)

    # ── Geo-temporal ───────────────────────────────────────────────────────
    impossible_travel_flag = int(ev.get('impossible_travel_flag', False))
    geo_time_diff_min      = float(ev.get('geo_time_diff_min', 999) or 999)

    # ── UNSW-NB15 network features ─────────────────────────────────────────
    spkts    = float(ev.get('spkts', 0) or 0)
    dpkts    = float(ev.get('dpkts', 0) or 0)
    sttl     = float(ev.get('sttl', 64) or 64)
    dttl     = float(ev.get('dttl', 64) or 64)
    sload    = float(ev.get('sload', 0) or 0)
    dload    = float(ev.get('dload', 0) or 0)
    sloss    = float(ev.get('sloss', 0) or 0)
    dloss    = float(ev.get('dloss', 0) or 0)
    sjit     = float(ev.get('sjit', 0) or 0)
    djit     = float(ev.get('djit', 0) or 0)
    swin     = float(ev.get('swin', 255) or 255)
    dwin     = float(ev.get('dwin', 255) or 255)
    smean    = float(ev.get('smean', sbytes / max(spkts, 1)))
    dmean    = float(ev.get('dmean', dbytes / max(dpkts, 1)))
    ct_srv_src    = float(ev.get('ct_srv_src', 0) or 0)
    ct_state_ttl  = float(ev.get('ct_state_ttl', 0) or 0)
    ct_dst_ltm    = float(ev.get('ct_dst_ltm', 0) or 0)
    ct_src_ltm    = float(ev.get('ct_src_ltm', 0) or 0)
    is_sm_ips_ports   = int(ev.get('is_sm_ips_ports', False))
    trans_depth       = float(ev.get('trans_depth', 0) or 0)
    response_body_len = float(ev.get('response_body_len', payload_sz) or 0)

    # ── Master lookup ──────────────────────────────────────────────────────
    fv = {
        # NETWORK
        'dur': dur, 'proto': proto_enc, 'service': service_enc, 'state': state_enc,
        'spkts': spkts, 'dpkts': dpkts, 'sbytes': sbytes, 'dbytes': dbytes,
        'rate': rate, 'sttl': sttl, 'dttl': dttl, 'sload': sload, 'dload': dload,
        'sloss': sloss, 'dloss': dloss, 'sjit': sjit, 'djit': djit,
        'swin': swin, 'dwin': dwin, 'smean': smean, 'dmean': dmean,
        'ct_srv_src': ct_srv_src, 'ct_state_ttl': ct_state_ttl,
        'ct_dst_ltm': ct_dst_ltm, 'ct_src_ltm': ct_src_ltm,
        'is_sm_ips_ports': is_sm_ips_ports, 'trans_depth': trans_depth,
        'response_body_len': response_body_len,
        # ENDPOINT
        'hour_of_day': hour_of_day, 'is_weekend': is_weekend,
        'process_encoded': process_encoded, 'parent_encoded': parent_encoded,
        'bytes_sent_log': bytes_sent_log, 'dst_port': dst_port,
        'is_external_dst': is_external_dst, 'username_encoded': username_encoded,
        'event_type_encoded': event_type_encoded,
        'failed_login_rate_1min': failed_login_rate_1min,
        'unique_dst_ips_10min': unique_dst_ips_10min,
        'spawn_depth': spawn_depth, 'anomaly_prob': anomaly_prob,
        # APPLICATION
        'is_night': is_night, 'payload_log': payload_log,
        'resp_time_log': resp_time_log, 'is_error': is_error,
        'is_server_err': is_server_err, 'is_large_resp': is_large_resp,
        'is_post': is_post, 'is_get': is_get,
        'has_known_attack_agent': has_known_attack_agent,
        'has_ambiguous_agent': has_ambiguous_agent,
        'has_browser_agent': has_browser_agent,
        'has_sqli_pattern': has_sqli_pattern,
        'endpoint_has_query': endpoint_has_query,
        'endpoint_query_length': endpoint_query_length,
        'is_admin_endpoint': is_admin_endpoint, 'is_auth_endpoint': is_auth_endpoint,
        'is_internal_ip': is_internal_ip, 'is_foreign_country': is_foreign_country,
        'country_enc': country_enc, 'method_enc': method_enc,
        'fail_rate_cumulative': fail_rate_cumulative, 'src_ip_req_log': src_ip_req_log,
        'impossible_travel_flag': impossible_travel_flag,
        'geo_time_diff_min': geo_time_diff_min,
    }

    row = []
    for feat in features:
        val = fv.get(feat, 0.0)
        try:
            row.append(float(val))
        except (TypeError, ValueError):
            row.append(0.0)
    return np.array([row])


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load label encoders saved by training scripts
    try:
        le_country = joblib.load('models/application/le_geo_country.pkl')
    except FileNotFoundError:
        from sklearn.preprocessing import LabelEncoder
        le_country = LabelEncoder()
        le_country.classes_ = np.array(['BR', 'CN', 'DE', 'FR', 'GB', 'IN', 'IR', 'JP', 'KP', 'KR', 'NG', 'RU', 'UA', 'US', 'unknown'])

    try:
        le_method = joblib.load('models/application/le_method.pkl')
    except FileNotFoundError:
        from sklearn.preprocessing import LabelEncoder
        le_method = LabelEncoder()
        le_method.classes_ = np.array(['DELETE', 'GET', 'PATCH', 'POST', 'PUT'])

    app.state.aux = {'le_country': le_country, 'le_method': le_method}

    app.state.models = {
        'network': {
            'binary':   joblib.load('models/network/lgbm_binary.pkl'),
            'features': joblib.load('models/network/features.pkl'),
        },
        'endpoint': {
            'classifier': joblib.load('models/endpoint/lgbm_classifier.pkl'),
            'iso_forest': joblib.load('models/endpoint/isolation_forest.pkl'),
            'scaler':     joblib.load('models/endpoint/scaler.pkl'),
            'features':   joblib.load('models/endpoint/features.pkl'),
        },
        'application': {
            'classifier': joblib.load('models/application/lgbm_classifier.pkl'),
            'features':   joblib.load('models/application/features.pkl'),
        }
    }

    # Fix endpoint scaler/classifier feature mismatch
    ep_feats  = app.state.models['endpoint']['features']
    ep_scaler = app.state.models['endpoint']['scaler']
    if ep_scaler.n_features_in_ != len(ep_feats):
        # Scaler was trained on 12 features (without anomaly_prob)
        # Classifier was trained on 13 features (with anomaly_prob)
        # Keep both lists separately so we scale the right subset
        scaler_features     = [f for f in ep_feats if f != 'anomaly_prob']
        classifier_features = ep_feats
        if len(scaler_features) != ep_scaler.n_features_in_:
            # anomaly_prob wasn't the culprit — just slice
            scaler_features     = ep_feats[:ep_scaler.n_features_in_]
            classifier_features = ep_feats
        app.state.models['endpoint']['scaler_features']     = scaler_features
        app.state.models['endpoint']['classifier_features'] = classifier_features
        app.state.models['endpoint']['features']            = classifier_features
        print(f"⚠️  Endpoint: scaler={len(scaler_features)} features, classifier={len(classifier_features)} features")
    else:
        app.state.models['endpoint']['scaler_features']     = ep_feats
        app.state.models['endpoint']['classifier_features'] = ep_feats

    print("✅ All models loaded.")
    yield
    print("🛑 Shutting down...")


app = FastAPI(title="Threat Detection Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

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
    is_external_dst: Optional[bool] = False
    spawn_depth: Optional[int] = 0
    unique_dst_ips_10min: Optional[float] = 0
    failed_login_rate_60s: Optional[float] = 0


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
    resp_time_ms: Optional[float] = 0
    src_ip_req_count: Optional[int] = 1
    fail_rate_cumulative: Optional[float] = 0
    impossible_travel_flag: Optional[bool] = False
    geo_time_diff_min: Optional[float] = 999


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ingest/network")
async def ingest_network(event: NetworkEvent):
    return await _process_event(event.dict(), 'network')

@app.post("/ingest/endpoint")
async def ingest_endpoint(event: EndpointEvent):
    return await _process_event(event.dict(), 'endpoint')

@app.post("/ingest/application")
async def ingest_app(event: AppEvent):
    return await _process_event(event.dict(), 'application')


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def _process_event(event: dict, layer: str):
    start = time.time()

    rule_results = rule_engine.evaluate_rules(event)
    ml_prob      = _ml_predict(event, layer)

    try:
        models   = app.state.models[layer]
        features = models['features']
        X        = encode_event(event, features, app.state.aux)
        shap_vals = explainer.get_shap_explanation(layer,X, features)
    except Exception as e:
        print(f"SHAP error: {e}")
        shap_vals = []
    shap_text = explainer.shap_to_english(shap_vals)

    layer_scores = {
        layer: ml_prob,
        **{l: None for l in ('network', 'endpoint', 'application') if l != layer}
    }

    fusion_result = fusion.compute_fusion_score(
        rule_results, ml_prob, layer_scores, event, shap_vals
    )

    playbook = None
    if fusion_result.severity in ('High', 'Critical'):
        playbook = explainer.generate_playbook({
            'attack_cat':      fusion_result.attack_cat,
            'severity':        fusion_result.severity,
            'mitre_tag':       fusion_result.mitre_tag,
            'score':           fusion_result.score,
            'layer_count':     fusion_result.layer_count,
            'src_ip':          event.get('src_ip'),
            'is_fp_candidate': fusion_result.is_fp_candidate,
        }, shap_text)

    return {
        "incident": {
            "score":               fusion_result.score,
            "severity":            fusion_result.severity,
            "attack_cat":          fusion_result.attack_cat,
            "mitre_tag":           fusion_result.mitre_tag,
            "rule_fired":          fusion_result.rule_fired,
            "ml_probability":      fusion_result.ml_probability,
            "layer_count":         fusion_result.layer_count,
            "is_fp_candidate":     fusion_result.is_fp_candidate,
            "raw_score":           getattr(fusion_result, 'raw_score', fusion_result.score),
            "fp_capped":           getattr(fusion_result, 'fp_capped', False),
            "cross_layer_boosted": getattr(fusion_result, 'cross_layer_boosted', False),
            "layer_boost":         getattr(fusion_result, 'layer_boost', 0),
            "active_layers":       [layer],
        },
        "explanation": {
            "shap_text":    shap_text,
            "top_features": [(f, round(v, 4)) for f, v in fusion_result.explanation_features],
            "rule_reasons": [r.reason for r in rule_results],
        },
        "playbook":      playbook,
        "processing_ms": round((time.time() - start) * 1000, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ML prediction
# ─────────────────────────────────────────────────────────────────────────────

def _ml_predict(event: dict, layer: str) -> float:
    try:
        models = app.state.models[layer]
        aux    = app.state.aux

        if layer == 'endpoint':
            scaler_feats     = models['scaler_features']
            classifier_feats = models['classifier_features']

            # Build and scale the 12-feature vector the scaler expects
            X_to_scale = encode_event(event, scaler_feats, aux)
            X_scaled   = models['scaler'].transform(X_to_scale)

            # Build full 13-feature vector for the classifier
            X_full = encode_event(event, classifier_feats, aux)

            # Substitute scaled values back in for the scaler columns
            for out_i, feat in enumerate(scaler_feats):
                in_i = classifier_feats.index(feat)
                X_full[0, in_i] = X_scaled[0, out_i]

            print(f"\n ML DEBUG — endpoint")
            print(f"  scaler features     : {len(scaler_feats)}")
            print(f"  classifier features : {len(classifier_feats)}")
            print(f"  non-zero (full vec) : {np.count_nonzero(X_full[0])}")
            nonzero_idx = np.nonzero(X_full[0])[0]
            for i in nonzero_idx[:10]:
                print(f"    {classifier_feats[i]:35s} = {X_full[0][i]:.4f}")

            prob = models['classifier'].predict_proba(X_full)[0][1]

        else:
            features = models['features']
            X = encode_event(event, features, aux)

            print(f"\n ML DEBUG — {layer}")
            print(f"  features expected : {len(features)}")
            print(f"  non-zero features : {np.count_nonzero(X[0])}")
            nonzero_idx = np.nonzero(X[0])[0]
            for i in nonzero_idx[:10]:
                print(f"    {features[i]:35s} = {X[0][i]:.4f}")

            if layer == 'network':
                prob = models['binary'].predict_proba(X)[0][1]
            elif layer == 'application':
                prob = models['classifier'].predict_proba(X)[0][1]
            else:
                prob = 0.0

        print(f"  ML probability    : {prob:.6f}")
        return float(prob)

    except Exception as e:
        import traceback
        print(f" ML ERROR ({layer}): {e}")
        traceback.print_exc()
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": hasattr(app.state, 'models')}