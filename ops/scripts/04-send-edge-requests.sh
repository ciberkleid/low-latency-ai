#!/usr/bin/env bash
set -euo pipefail

# Access endpoints to exercise client-side and server-side inference

# Swagger UI:
# open http://localhost:8080/swagger-ui/index.html

print_response_and_time() {
  response_file="$(mktemp)"
  time_output="$1"
  shift

  curl "$@" -sS -o "$response_file" -w "$time_output"

  if command -v jq >/dev/null 2>&1; then
    jq . "$response_file" 2>/dev/null || cat "$response_file"
  else
    cat "$response_file"
  fi

  rm -f "$response_file"
}

echo

echo "### Client-side inference"
print_response_and_time 'latency_total=%{time_total}s\n' -X 'POST' \
  'http://localhost:8080/ai/inference/checkSentiment' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d 'I love Spring'

echo
echo
