# Threat Detection Engine — Quickstart

## Prerequisites
```bash
pip install lightgbm scikit-learn shap pandas numpy fastapi uvicorn anthropic faker joblib
```

## Step-by-step

### 1. Download UNSW-NB15 (network layer)
Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Download and place:
- `data/raw/UNSW_NB15_training-set.csv`
- `data/raw/UNSW_NB15_testing-set.csv`

### 2. Generate synthetic logs
```bash
python utils/generate_endpoint_logs.py   # ~1 min
python utils/generate_app_logs.py        # ~1 min
```

### 3. Train all models
```bash
python models/network/train_network_model.py       # ~5-10 min
python models/endpoint/train_endpoint_model.py     # ~3-5 min
python models/application/train_app_model.py       # ~3-5 min
```
Each script saves .pkl files in the corresponding models/ subfolder.

### 4. Run the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Test with a sample event
```bash
# Brute force attempt
curl -X POST http://localhost:8000/ingest/network \
  -H "Content-Type: application/json" \
  -d '{"src_ip":"1.2.3.4","dst_ip":"10.0.0.5","proto":"tcp","sbytes":800,"dbytes":200,"rate":15,"dur":0.1,"failed_login_rate_60s":9}'

# SQL injection
curl -X POST http://localhost:8000/ingest/application \
  -H "Content-Type: application/json" \
  -d '{"src_ip":"5.6.7.8","user_agent":"sqlmap/1.7","method":"GET","endpoint":"/api/users?id=1 OR 1=1--","status_code":500,"payload_size":300,"geo_country":"CN"}'
```

## Docker (later)
```bash
docker compose up --build
```
Models must be pre-trained. Mount the models/ folder as a volume.
