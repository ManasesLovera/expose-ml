# Expose ML

API server that expose ML models via HTTP.

## Configuration

Create `.env` from `.env.example` and set `MODEL_FILENAME` to one of the files inside `saved_models/`.

## Run

```bash
python main.py
```

## Predict

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text":"Free money now!!!"}'
```
