#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-http://localhost:8000}"

echo "Checking health endpoint..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HOST/health")
if [ "$HEALTH_STATUS" != "200" ]; then
  echo "FAIL: /health returned $HEALTH_STATUS"
  exit 1
fi
echo "OK: /health returned 200"

echo "Checking prediction endpoint..."
# tiny 1x1 red pixel JPEG, base64-decoded, just to prove /predict responds correctly
echo "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a
HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy
MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIA
AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEB
AQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX
/9k=" | base64 -d > /tmp/test_pixel.jpg

PREDICT_STATUS=$(curl -s -o /tmp/predict_response.json -w "%{http_code}" \
  -X POST "$HOST/predict" -F "file=@/tmp/test_pixel.jpg;type=image/jpeg")

if [ "$PREDICT_STATUS" != "200" ]; then
  echo "FAIL: /predict returned $PREDICT_STATUS"
  cat /tmp/predict_response.json
  exit 1
fi
echo "OK: /predict returned 200 -> $(cat /tmp/predict_response.json)"

echo "Smoke tests passed."