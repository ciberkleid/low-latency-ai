#!/usr/bin/env bash
set -euo pipefail

# Exercises the edge inference endpoint (client-side, in-JVM inference).
# Swagger UI: open http://localhost:8080/swagger-ui/index.html

TIME_FORMAT='\nlatency_total=%{time_total}s'

echo -e "\n### Edge inference — check sentiment"
curlie -w "$TIME_FORMAT" "http://localhost:8080/ai/inference/checkSentiment" \
  Content-Type:application/json \
  body='I love Spring'

echo