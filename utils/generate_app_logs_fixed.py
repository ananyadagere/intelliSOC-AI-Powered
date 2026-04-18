"""
FIXED Synthetic Application / HTTP Log Generator
================================================
The original had data leakage: features like 'has_sqli_pattern', 'is_bad_agent',
'impossible_travel', 'failed_count_1min' were computed from raw columns that
directly encode the label (e.g. sqlmap user_agent → is_bad_agent=1 → label=1).
This caused 100% accuracy because the model was basically reading the label.

Fix: These features are now computed at INFERENCE TIME from raw fields only,
NOT stored in the CSV. The CSV stores only raw observable fields.
The model is trained on features that a real system would compute at runtime.
"""
import pandas as pd
import numpy as np
import random
import re
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()
random.seed(99)
np.random.seed(99)

os.makedirs('data/synthetic', exist_ok=True)

START_TIME = datetime(2025, 1, 1, 0, 0, 0)
TOTAL_ROWS = 50_000

USER_IDS = [f"user_{i:04d}" for i in range(1, 300)]
ADMIN_IDS = ['admin_001', 'sysadmin', 'superuser']
INTERNAL_IPS = [f"10.0.{random.randint(0,5)}.{random.randint(1,254)}" for _ in range(100)]
EXTERNAL_IPS = [fake.ipv4_public() for _ in range(200)]

USER_HOME_COUNTRY = {uid: random.choice(['IN', 'US', 'GB', 'DE', 'JP']) for uid in USER_IDS}
FOREIGN_COUNTRIES = ['CN', 'RU', 'KP', 'IR', 'NG', 'UA']
HOME_COUNTRIES = ['IN', 'US', 'GB', 'DE', 'JP']

GOOD_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1',
]

# Subtly suspicious agents — realistic tools that aren't obviously attack agents
# These should NOT be a dead giveaway; the model must learn from behavior patterns
MIXED_AGENTS = [
    'python-requests/2.31.0',   # used for automation AND attacks
    'curl/7.88.1',               # used for scripts AND attacks
    'Go-http-client/1.1',        # used for microservices AND attacks
    'axios/1.6.0',               # used for frontend AND attacks
]

CLEAR_BAD_AGENTS = [
    'sqlmap/1.7.12#stable (https://sqlmap.org)',
    'Hydra v9.5',
    'Nikto/2.1.6',
    'Medusa v2.2',
    'masscan/1.3.2',
]

BENIGN_ENDPOINTS = [
    '/api/products', '/api/users/profile', '/api/orders', '/dashboard',
    '/api/search', '/api/cart', '/api/checkout', '/api/inventory',
    '/health', '/api/notifications', '/api/settings', '/api/reports',
    '/api/reviews', '/api/categories', '/api/recommendations',
]

# IMPORTANT: SQLi payloads go in endpoint URL but are NOT pre-labeled
# The model must learn to detect them from the raw endpoint string
SQLI_ENDPOINTS = [
    "/api/users?id=' OR '1'='1",
    "/search?q=' OR 1=1--",
    "/api/products?filter=UNION SELECT null,username,password FROM users--",
    "/api/orders?user='; DROP TABLE users;--",
    "/api/users?id=1; exec xp_cmdshell('whoami')--",
    "/api/search?q=' AND 1=2 UNION SELECT table_name FROM information_schema.tables--",
    "/login?user=admin'--",
    "/api/items?id=1' ORDER BY 3--",
    "/api/data?input=1%27%20OR%20%271%27%3D%271",  # URL encoded
]

def random_ts(start=START_TIME, days=7):
    return start + timedelta(seconds=random.randint(0, days * 86400))

def random_daytime_ts():
    """Returns a timestamp during business hours"""
    ts = random_ts()
    return ts.replace(hour=random.randint(8, 20))

def random_night_ts():
    """Returns a timestamp at night (attack scenario)"""
    ts = random_ts()
    return ts.replace(hour=random.randint(1, 5))

# ─── BENIGN ROWS ──────────────────────────────────────────────────────────────

def benign_row(ts=None):
    ts = ts or random_daytime_ts()
    uid = random.choice(USER_IDS)
    home = USER_HOME_COUNTRY[uid]
    return {
        'timestamp':        ts.isoformat(),
        'src_ip':           random.choice(INTERNAL_IPS + EXTERNAL_IPS[:50]),
        'user_agent':       random.choice(GOOD_AGENTS),
        'method':           random.choices(['GET', 'POST', 'PUT', 'DELETE'], weights=[60, 25, 10, 5])[0],
        'endpoint':         random.choice(BENIGN_ENDPOINTS),
        'status_code':      random.choices([200, 201, 204, 304], weights=[70, 15, 10, 5])[0],
        'payload_size':     random.randint(200, 5000),
        'response_time_ms': random.randint(50, 800),
        'geo_country':      home,
        'geo_city':         fake.city(),
        'session_id':       fake.uuid4(),
        'user_id':          uid,
        # IMPORTANT: these raw fields are what the model trains on
        # 'impossible_travel', 'sqli_pattern', etc are computed at runtime
        'label':            0,
        'attack_cat':       'Normal',
        'is_fp_candidate':  0,
    }

def admin_export_fp_row(ts=None):
    """False positive: admin doing large legitimate export during business hours"""
    ts = ts or random_daytime_ts()
    uid = random.choice(ADMIN_IDS)
    # Key distinguishing features vs real exfil:
    # - Internal src_ip, home country, business hours, admin user_id, status 200
    return {
        'timestamp':        ts.isoformat(),
        'src_ip':           random.choice(INTERNAL_IPS),   # internal = safe
        'user_agent':       random.choice(GOOD_AGENTS),
        'method':           'GET',
        'endpoint':         random.choice(['/api/export/report', '/api/backup', '/api/dump/logs']),
        'status_code':      200,
        'payload_size':     random.randint(8_000_000, 50_000_000),  # large but internal
        'response_time_ms': random.randint(3000, 30000),
        'geo_country':      random.choice(HOME_COUNTRIES),  # home country = safe
        'geo_city':         fake.city(),
        'session_id':       fake.uuid4(),
        'user_id':          uid,
        'label':            0,
        'attack_cat':       'Normal',
        'is_fp_candidate':  1,
    }

# ─── ATTACK ROWS ──────────────────────────────────────────────────────────────

def brute_force_rows(group_id):
    """Burst of failed logins - distinguishable by: status 401/403, high freq, bad agent"""
    src_ip   = random.choice(EXTERNAL_IPS[50:])
    endpoint = random.choice(['/api/login', '/admin/login', '/wp-login.php', '/api/auth'])
    ts       = random_ts()
    rows = []
    burst_count = random.randint(20, 60)
    for i in range(burst_count):
        # Mix: some use bad agents, some use ambiguous agents — not all are obvious
        agent = random.choices(
            CLEAR_BAD_AGENTS + MIXED_AGENTS,
            weights=[0.3]*5 + [0.425]*4  # 30% obvious, 70% ambiguous
        )[0]
        rows.append({
            'timestamp':        (ts + timedelta(seconds=i * random.uniform(0.5, 3))).isoformat(),
            'src_ip':           src_ip,
            'user_agent':       agent,
            'method':           'POST',
            'endpoint':         endpoint,
            'status_code':      random.choices([401, 403, 429], weights=[70, 20, 10])[0],
            'payload_size':     random.randint(50, 500),
            'response_time_ms': random.randint(100, 500),
            'geo_country':      random.choice(FOREIGN_COUNTRIES),
            'geo_city':         fake.city(),
            'session_id':       None,
            'user_id':          None,  # no valid user
            'label':            1,
            'attack_cat':       'Brute Force',
            'is_fp_candidate':  0,
        })
    return rows

def sqli_rows(group_id):
    """SQL injection - distinguishable by: malformed endpoint, error 500, specific agents"""
    src_ip = random.choice(EXTERNAL_IPS[100:])
    ts = random_ts()
    rows = []
    count = random.randint(5, 15)
    for i in range(count):
        endpoint = random.choice(SQLI_ENDPOINTS)
        rows.append({
            'timestamp':        (ts + timedelta(seconds=i * random.uniform(2, 10))).isoformat(),
            'src_ip':           src_ip,
            # Mix of obvious and subtle agents
            'user_agent':       random.choices(
                                    CLEAR_BAD_AGENTS[:1] + MIXED_AGENTS,
                                    weights=[0.4, 0.15, 0.15, 0.15, 0.15]
                                )[0],
            'method':           random.choice(['GET', 'POST']),
            'endpoint':         endpoint,  # raw SQLi in URL — model learns from this
            'status_code':      random.choices([500, 200, 400, 403], weights=[50, 25, 15, 10])[0],
            'payload_size':     random.randint(100, 2000),
            'response_time_ms': random.randint(50, 3000),  # errors are fast or slow
            'geo_country':      random.choice(FOREIGN_COUNTRIES),
            'geo_city':         fake.city(),
            'session_id':       None,
            'user_id':          None,
            'label':            1,
            'attack_cat':       'SQLi',
            'is_fp_candidate':  0,
        })
    return rows

def impossible_travel_rows(group_id):
    """Stolen credentials: same user_id from two different countries within 60 min"""
    uid          = random.choice(USER_IDS)
    home_country = USER_HOME_COUNTRY[uid]
    # Pick a foreign country that is geographically impossible to travel to in 30 min
    foreign      = random.choice(FOREIGN_COUNTRIES)
    ts           = random_ts()
    rows = []

    # First login: legitimate, home country
    rows.append({
        'timestamp':        ts.isoformat(),
        'src_ip':           random.choice(EXTERNAL_IPS[:50]),
        'user_agent':       random.choice(GOOD_AGENTS),
        'method':           'POST',
        'endpoint':         '/api/login',
        'status_code':      200,
        'payload_size':     random.randint(200, 500),
        'response_time_ms': random.randint(100, 400),
        'geo_country':      home_country,
        'geo_city':         fake.city(),
        'session_id':       fake.uuid4(),
        'user_id':          uid,
        'label':            0,  # this specific login is normal
        'attack_cat':       'Normal',
        'is_fp_candidate':  0,
    })

    # Follow-up requests from foreign country within 5-30 minutes
    offset_min = random.randint(5, 30)
    for i in range(random.randint(4, 10)):
        rows.append({
            'timestamp':        (ts + timedelta(minutes=offset_min, seconds=i*45)).isoformat(),
            'src_ip':           random.choice(EXTERNAL_IPS[150:]),
            'user_agent':       random.choice(GOOD_AGENTS),  # valid UA — harder to detect
            'method':           random.choices(['GET', 'POST'], weights=[60, 40])[0],
            'endpoint':         random.choice(['/api/users/profile', '/api/export/report',
                                               '/api/orders', '/api/settings']),
            'status_code':      200,  # succeeds because token is valid
            'payload_size':     random.randint(500, 80000),
            'response_time_ms': random.randint(200, 1200),
            'geo_country':      foreign,   # different country from login above
            'geo_city':         fake.city(),
            'session_id':       fake.uuid4(),
            'user_id':          uid,       # same user_id — the tell
            'label':            1,
            'attack_cat':       'Account Takeover',
            'is_fp_candidate':  0,
        })
    return rows

def http_exfil_rows(group_id):
    """Data exfil: large downloads at night from external/foreign IP"""
    uid    = random.choice(USER_IDS)
    ts     = random_night_ts()  # night hours = suspicious
    src_ip = random.choice(EXTERNAL_IPS[100:])
    rows = []
    for i in range(random.randint(5, 12)):
        rows.append({
            'timestamp':        (ts + timedelta(minutes=i * 15)).isoformat(),
            'src_ip':           src_ip,
            'user_agent':       random.choice(GOOD_AGENTS + MIXED_AGENTS),
            'method':           'GET',
            'endpoint':         random.choice(['/api/export', '/api/backup',
                                               '/api/dump', '/api/users/all']),
            'status_code':      200,
            'payload_size':     random.randint(5_000_000, 100_000_000),  # huge = suspicious
            'response_time_ms': random.randint(5000, 60000),
            'geo_country':      random.choice(FOREIGN_COUNTRIES),
            'geo_city':         fake.city(),
            'session_id':       fake.uuid4(),
            'user_id':          uid,
            'label':            1,
            'attack_cat':       'Data Exfil',
            'is_fp_candidate':  0,
        })
    return rows

# ─── ASSEMBLE ─────────────────────────────────────────────────────────────────

rows = []

n_benign = int(TOTAL_ROWS * 0.70) - 500
for _ in range(n_benign):
    rows.append(benign_row())

for _ in range(500):
    rows.append(admin_export_fp_row())

group = 0
while sum(1 for r in rows if r['attack_cat'] == 'Brute Force') < int(TOTAL_ROWS * 0.10):
    rows.extend(brute_force_rows(f'BF_{group}'))
    group += 1

group = 0
while sum(1 for r in rows if r['attack_cat'] == 'SQLi') < int(TOTAL_ROWS * 0.10):
    rows.extend(sqli_rows(f'SQLI_{group}'))
    group += 1

group = 0
while sum(1 for r in rows if r['attack_cat'] == 'Account Takeover') < int(TOTAL_ROWS * 0.05):
    rows.extend(impossible_travel_rows(f'IT_{group}'))
    group += 1

group = 0
while sum(1 for r in rows if r['attack_cat'] == 'Data Exfil') < int(TOTAL_ROWS * 0.05):
    rows.extend(http_exfil_rows(f'EX_{group}'))
    group += 1

random.shuffle(rows)
rows = rows[:TOTAL_ROWS]

df = pd.DataFrame(rows)
df = df.sort_values('timestamp').reset_index(drop=True)

out_path = 'data/synthetic/app_logs.csv'
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} app log rows -> {out_path}")
print("\nClass distribution:")
print(df['attack_cat'].value_counts())
print(f"\nFalse positive candidates (admin exports): {df['is_fp_candidate'].sum()}")
print("\nIMPORTANT: Re-run train_app_model.py after regenerating this data.")
