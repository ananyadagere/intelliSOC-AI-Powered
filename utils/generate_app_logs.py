"""
Synthetic Application / HTTP Log Generator
Run: python generate_app_logs.py
Outputs: data/synthetic/app_logs.csv (~50,000 rows)
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

# ─── Baseline data ─────────────────────────────────────────────────────────

USER_IDS = [f"user_{i:04d}" for i in range(1, 300)]
ADMIN_IDS = ['admin_001', 'sysadmin', 'superuser']
INTERNAL_IPS = [f"10.0.{random.randint(0,5)}.{random.randint(1,254)}" for _ in range(100)]
EXTERNAL_IPS = [fake.ipv4_public() for _ in range(200)]

# Assign each user to a home country (for impossible travel detection)
USER_HOME_COUNTRY = {uid: random.choice(['IN', 'US', 'GB', 'DE', 'JP']) for uid in USER_IDS}
FOREIGN_COUNTRIES = ['CN', 'RU', 'KP', 'IR', 'NG', 'UA']

GOOD_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1',
]

BAD_AGENTS = [
    'sqlmap/1.7.12#stable (https://sqlmap.org)',
    'python-requests/2.31.0',
    'Hydra v9.5',
    'Nikto/2.1.6',
    'Medusa v2.2',
    'masscan/1.3.2',
    'curl/7.88.1',
    'Go-http-client/1.1',
]

BENIGN_ENDPOINTS = [
    '/api/products', '/api/users/profile', '/api/orders', '/dashboard',
    '/api/search', '/api/cart', '/api/checkout', '/api/inventory',
    '/health', '/api/notifications', '/api/settings', '/api/reports',
]

SQLI_PAYLOADS = [
    "' OR '1'='1", "' OR 1=1--", "UNION SELECT null,username,password FROM users--",
    "'; DROP TABLE users;--", "1; exec xp_cmdshell('whoami')--",
    "' AND 1=2 UNION SELECT table_name FROM information_schema.tables--",
    "admin'--", "' OR 'x'='x", "1' ORDER BY 3--",
]

# ─── Row generators ────────────────────────────────────────────────────────

def random_ts(start=START_TIME, days=7):
    return start + timedelta(seconds=random.randint(0, days * 86400))

def benign_row(ts=None):
    ts = ts or random_ts()
    uid = random.choice(USER_IDS)
    return {
        'timestamp': ts,
        'src_ip': random.choice(INTERNAL_IPS + EXTERNAL_IPS[:50]),
        'user_agent': random.choice(GOOD_AGENTS),
        'method': random.choices(['GET', 'POST', 'PUT', 'DELETE'], weights=[60, 25, 10, 5])[0],
        'endpoint': random.choice(BENIGN_ENDPOINTS),
        'status_code': random.choices([200, 201, 204, 304], weights=[70, 15, 10, 5])[0],
        'payload_size': random.randint(200, 5000),
        'response_time_ms': random.randint(50, 800),
        'geo_country': USER_HOME_COUNTRY[uid],
        'geo_city': fake.city(),
        'session_id': fake.uuid4(),
        'user_id': uid,
        'auth_header': f'Bearer {fake.sha256()[:32]}',
        'label': 0,
        'attack_cat': 'Normal',
        'is_fp_candidate': 0,
        'attack_group_id': None
    }

def admin_export_fp_row(ts=None):
    """False positive: admin legitimate large export"""
    ts = ts or random_ts()
    uid = random.choice(ADMIN_IDS)
    return {
        'timestamp': ts,
        'src_ip': random.choice(INTERNAL_IPS),
        'user_agent': random.choice(GOOD_AGENTS),
        'method': 'GET',
        'endpoint': random.choice(['/api/export/report', '/api/backup', '/api/dump/logs']),
        'status_code': 200,
        'payload_size': random.randint(10_000_000, 50_000_000),  # 10-50MB
        'response_time_ms': random.randint(3000, 30000),
        'geo_country': 'IN',  # home country
        'geo_city': 'Bengaluru',
        'session_id': fake.uuid4(),
        'user_id': uid,
        'auth_header': f'Bearer admin-{fake.sha256()[:24]}',
        'label': 0,
        'attack_cat': 'Normal',
        'is_fp_candidate': 1,
        'attack_group_id': None
    }

def brute_force_rows(group_id):
    """Repeated failed logins from same IP"""
    src_ip = random.choice(EXTERNAL_IPS[50:])
    endpoint = random.choice(['/api/login', '/admin/login', '/wp-login.php', '/api/auth'])
    ts = random_ts()
    rows = []
    for i in range(random.randint(20, 60)):
        rows.append({
            'timestamp': ts + timedelta(seconds=i * random.uniform(1, 4)),
            'src_ip': src_ip,
            'user_agent': random.choice(BAD_AGENTS[:4]),
            'method': 'POST',
            'endpoint': endpoint,
            'status_code': random.choices([401, 403, 429], weights=[70, 20, 10])[0],
            'payload_size': random.randint(50, 500),
            'response_time_ms': random.randint(100, 500),
            'geo_country': random.choice(FOREIGN_COUNTRIES),
            'geo_city': fake.city(),
            'session_id': None,
            'user_id': None,
            'auth_header': None,
            'label': 1,
            'attack_cat': 'Brute Force',
            'is_fp_candidate': 0,
            'attack_group_id': group_id
        })
    return rows

def sqli_rows(group_id):
    """SQL injection attempts"""
    src_ip = random.choice(EXTERNAL_IPS[100:])
    ts = random_ts()
    rows = []
    for i, payload in enumerate(random.choices(SQLI_PAYLOADS, k=random.randint(5, 15))):
        endpoint = random.choice([
            f"/api/users?id={payload}",
            f"/search?q={payload}",
            f"/api/products?filter={payload}",
            f"/api/orders?user={payload}"
        ])
        rows.append({
            'timestamp': ts + timedelta(seconds=i * random.uniform(2, 10)),
            'src_ip': src_ip,
            'user_agent': random.choice(['sqlmap/1.7.12#stable', 'python-requests/2.31.0']),
            'method': random.choice(['GET', 'POST']),
            'endpoint': endpoint,
            'status_code': random.choices([500, 200, 400], weights=[50, 30, 20])[0],
            'payload_size': random.randint(100, 2000),
            'response_time_ms': random.randint(50, 2000),
            'geo_country': random.choice(FOREIGN_COUNTRIES),
            'geo_city': fake.city(),
            'session_id': None,
            'user_id': None,
            'auth_header': None,
            'label': 1,
            'attack_cat': 'SQLi',
            'is_fp_candidate': 0,
            'attack_group_id': group_id
        })
    return rows

def impossible_travel_rows(group_id):
    """Same user logs in from home country, then foreign country < 60min"""
    uid = random.choice(USER_IDS)
    home_country = USER_HOME_COUNTRY[uid]
    foreign_country = random.choice(FOREIGN_COUNTRIES)
    ts = random_ts()
    token = f'Bearer stolen-{fake.sha256()[:24]}'
    
    rows = []
    # Legitimate login (home)
    rows.append({
        'timestamp': ts,
        'src_ip': random.choice(EXTERNAL_IPS[:50]),
        'user_agent': random.choice(GOOD_AGENTS),
        'method': 'POST',
        'endpoint': '/api/login',
        'status_code': 200,
        'payload_size': random.randint(200, 500),
        'response_time_ms': random.randint(100, 400),
        'geo_country': home_country,
        'geo_city': fake.city(),
        'session_id': fake.uuid4(),
        'user_id': uid,
        'auth_header': token,
        'label': 0,
        'attack_cat': 'Normal',
        'is_fp_candidate': 0,
        'attack_group_id': group_id
    })
    
    # Suspicious login (foreign, <30min later)
    for i in range(random.randint(3, 8)):
        rows.append({
            'timestamp': ts + timedelta(minutes=random.randint(5, 30), seconds=i*60),
            'src_ip': random.choice(EXTERNAL_IPS[150:]),
            'user_agent': random.choice(GOOD_AGENTS),  # valid agent — stolen credentials
            'method': random.choices(['GET', 'POST'], weights=[60, 40])[0],
            'endpoint': random.choice(['/api/users/profile', '/api/export/report', '/api/orders']),
            'status_code': 200,
            'payload_size': random.randint(500, 50000),
            'response_time_ms': random.randint(200, 1000),
            'geo_country': foreign_country,
            'geo_city': fake.city(),
            'session_id': fake.uuid4(),
            'user_id': uid,
            'auth_header': token,  # same token = stolen
            'label': 1,
            'attack_cat': 'Account Takeover',
            'is_fp_candidate': 0,
            'attack_group_id': group_id
        })
    return rows

def http_exfil_rows(group_id):
    """Large repeated data downloads at night"""
    uid = random.choice(USER_IDS)
    night_ts = random_ts() + timedelta(hours=2)  # early morning
    night_ts = night_ts.replace(hour=random.randint(1, 5))
    src_ip = random.choice(EXTERNAL_IPS[100:])
    rows = []
    for i in range(random.randint(5, 12)):
        rows.append({
            'timestamp': night_ts + timedelta(minutes=i * 15),
            'src_ip': src_ip,
            'user_agent': random.choice(GOOD_AGENTS),
            'method': 'GET',
            'endpoint': random.choice(['/api/export', '/api/backup', '/api/dump', '/api/users/all']),
            'status_code': 200,
            'payload_size': random.randint(5_000_000, 100_000_000),  # 5-100MB
            'response_time_ms': random.randint(5000, 60000),
            'geo_country': random.choice(FOREIGN_COUNTRIES),
            'geo_city': fake.city(),
            'session_id': fake.uuid4(),
            'user_id': uid,
            'auth_header': f'Bearer {fake.sha256()[:32]}',
            'label': 1,
            'attack_cat': 'Data Exfil',
            'is_fp_candidate': 0,
            'attack_group_id': group_id
        })
    return rows

# ─── Assemble dataset ──────────────────────────────────────────────────────

rows = []

# Benign (normal traffic)
n_benign = int(TOTAL_ROWS * 0.70) - 500
for _ in range(n_benign):
    rows.append(benign_row())

# Admin export false positives
for _ in range(500):
    rows.append(admin_export_fp_row())

# Attack scenarios (two simultaneous: brute force + C2/impossible travel)
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

print(f"✅ Generated {len(df)} app log rows → {out_path}")
print("\nClass distribution:")
print(df['attack_cat'].value_counts())
print(f"\nFalse positive candidates (admin exports): {df['is_fp_candidate'].sum()}")
