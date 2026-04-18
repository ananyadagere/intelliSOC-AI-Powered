# scoring/explainer.py

import shap
import numpy as np
import joblib
import requests


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


def _is_ollama_running() -> bool:
    try:
        r = requests.get("http://127.0.0.1:11434", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def get_shap_explanation(model_name: str, X: np.ndarray, feature_list: list) -> list[tuple]:
    """
    Returns top-5 (feature, shap_value) tuples.
    X must be the already-encoded numpy array (shape 1 x n_features).
    feature_list must match the columns of X in order.
    """
    try:
        explainer = joblib.load(f'models/{model_name}/shap_explainer.pkl')
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1 for binary classifier
        return list(zip(feature_list, shap_vals[0]))
    except Exception as e:
        print(f"SHAP error: {e}")
        return []


def shap_to_english(shap_features: list[tuple]) -> str:
    """Convert SHAP values to plain English bullets."""
    lines = []
    for feat, val in sorted(shap_features, key=lambda x: abs(x[1]), reverse=True)[:5]:
        direction = "increased" if val > 0 else "decreased"
        readable  = FEATURE_EXPLANATIONS.get(feat, feat.replace('_', ' '))
        lines.append(f"• {readable} {direction} the threat score by {abs(val):.3f}")
    return "\n".join(lines)


def generate_playbook(incident_context: dict, shap_explanation: str) -> str:
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

Generate a concise, actionable incident response playbook with:
1. Immediate Actions
2. Investigation Steps
3. Containment
4. False Positive Check
5. Prevention
"""

    if not _is_ollama_running():
        print("[explainer] Ollama not detected — using static playbooks")
        return _static_playbook(incident_context)

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"[explainer] Ollama request failed: {e}")
        return _static_playbook(incident_context)


def _static_playbook(ctx: dict) -> str:
    cat      = ctx.get('attack_cat', 'Unknown')
    severity = ctx.get('severity', 'Unknown')
    src_ip   = ctx.get('src_ip', 'N/A')
    mitre    = ctx.get('mitre_tag', 'N/A')

    return f"""## Incident Response Playbook ({severity} — {cat})

**MITRE ATT&CK:** {mitre}  
**Source IP:** {src_ip}

### 1. Immediate Actions
- Verify the alert is not a false positive by cross-checking with other log sources.
- Escalate to Tier-2 analyst if severity is High or Critical.
- Tag the source IP for enhanced monitoring.

### 2. Investigation Steps
- Pull full session logs for {src_ip} from the last 24 hours.
- Check for lateral movement from this source.
- Correlate with endpoint and application layer events.

### 3. Containment
- If confirmed malicious: block {src_ip} at the perimeter firewall.
- Isolate any affected endpoints from the network.
- Revoke active sessions tied to the source.

### 4. False Positive Check
- Confirm whether {src_ip} is a known scanner, pentest tool, or CI/CD agent.
- Check if the behaviour matches a scheduled job or data export.
- Review user/asset context in CMDB.

### 5. Prevention
- Tune detection rules to reduce noise for this pattern.
- Add {src_ip} to watchlist if behaviour repeats.
- Review and harden the exposed service if applicable.
"""