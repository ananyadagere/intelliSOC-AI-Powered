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