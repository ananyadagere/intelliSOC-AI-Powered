# scoring/fusion.py

from dataclasses import dataclass
from typing import Optional
from typing import List, Tuple

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
    explanation_features: List[Tuple[str, float]]  # SHAP features

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