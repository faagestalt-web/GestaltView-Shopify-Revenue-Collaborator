# Usage Examples (Python + Shell)

This repo includes a Node/React Shopify app. Below are **copy-paste-friendly** examples for calling the backend API from **Python** and **shell** scripts so you can quickly validate the endpoints locally.

## Prerequisites

- Start the backend server (default `http://localhost:3000`).
- Optional: start the frontend dev server on `http://localhost:5173`.

## Python examples

Save as `scripts/api_examples.py` (or run inline) and adjust URLs as needed.

```python
import json
import requests

BASE_URL = "http://localhost:3000"

# 1) Capture a bucket drop
bucket_payload = {"text": "New product idea: seasonal gift bundles", "tags": ["idea", "seasonal"]}
bucket_res = requests.post(f"{BASE_URL}/api/collaborator/bucket", json=bucket_payload)
print("Bucket drop:", bucket_res.json())

# 2) Generate an artifact
generate_payload = {"type": "story", "topic": "How our brand started"}
generate_res = requests.post(f"{BASE_URL}/api/collaborator/generate", json=generate_payload)
print("Generated artifact:", json.dumps(generate_res.json(), indent=2))

# 3) Analyze resonance
analyze_payload = {"content": "We create timeless pieces for intentional living."}
analyze_res = requests.post(f"{BASE_URL}/api/collaborator/analyze", json=analyze_payload)
print("Analysis:", json.dumps(analyze_res.json(), indent=2))

# 4) Improve text
improve_payload = {"content": "We make products people love."}
improve_res = requests.post(f"{BASE_URL}/api/collaborator/improve", json=improve_payload)
print("Improved:", json.dumps(improve_res.json(), indent=2))
```

> Note: Install Python dependencies with `pip install requests` if needed.

## Shell script examples (curl)

Save as `scripts/api_examples.sh` and run with `bash scripts/api_examples.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://localhost:3000"

echo "1) Bucket drop"
curl -sS "${BASE_URL}/api/collaborator/bucket" \
  -H "Content-Type: application/json" \
  -d '{"text":"Homepage headline idea: made for the bold","tags":["copy","headline"]}' \
  | jq .

echo "2) Generate artifact"
curl -sS "${BASE_URL}/api/collaborator/generate" \
  -H "Content-Type: application/json" \
  -d '{"type":"story","topic":"Our sustainability journey"}' \
  | jq .

echo "3) Analyze resonance"
curl -sS "${BASE_URL}/api/collaborator/analyze" \
  -H "Content-Type: application/json" \
  -d '{"content":"We help creators build meaningful brands."}' \
  | jq .

echo "4) Improve text"
curl -sS "${BASE_URL}/api/collaborator/improve" \
  -H "Content-Type: application/json" \
  -d '{"content":"We make things that matter."}' \
  | jq .
```

> Note: Install `jq` for pretty-printed JSON output (`brew install jq` on macOS).
