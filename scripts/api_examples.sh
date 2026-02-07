#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${GESTALTVIEW_API_URL:-http://localhost:3000}"

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
