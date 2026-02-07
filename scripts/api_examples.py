import json
import os
import requests

BASE_URL = os.getenv("GESTALTVIEW_API_URL", "http://localhost:3000")

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
