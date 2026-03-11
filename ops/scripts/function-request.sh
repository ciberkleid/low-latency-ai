#!/usr/bin/env bash
set -euo pipefail

# Exercises the data-local inference endpoint (server-side, GemFire function execution).
# Swagger UI: open http://localhost:8080/swagger-ui/index.html

TIME_FORMAT='\nlatency_total=%{time_total}s'

echo -e "\n### Function inference — product review sentiment for Lawnmower"
curlie -w "$TIME_FORMAT" "http://localhost:8080/product/review/Lawnmower"

#echo -e "\n\n### Function inference — product review sentiment for Coffee Mug"
#curlie -w "$TIME_FORMAT" "http://localhost:8080/product/review/Coffee%20Mug"
#
#echo -e "\n\n### Function inference — product review sentiment for Lawnmower"
#curlie -w "$TIME_FORMAT" "http://localhost:8080/product/review/Lawnmower"

echo