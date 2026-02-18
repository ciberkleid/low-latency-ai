#!/usr/bin/env bash
set -euo pipefail

# Require Java 21 for compatibility with GemFire.
JAVA_MAJOR="$(java -version 2>&1 | awk -F '[\".]' '/version/ {print $2; exit}')"
if [[ "$JAVA_MAJOR" != "21" ]]; then
  echo "Java 21 is required for compatibility with GemFire. Current Java major version: ${JAVA_MAJOR:-unknown}." >&2
  echo "Set Java to 21 and re-run. Example: sdk install java 21.0.10-librca && sdk use java 21.0.10-librca" >&2
  exit 1
fi

# Build java components (shared domain, client app, server-side function)
./mvnw -pl libraries/shared-domain -am -DskipTests install # install shared domain library
./mvnw -DskipTests package  # build app and function

# Deploy server-side function
docker cp functions/inference-function/target/inference-function-0.0.1-SNAPSHOT.jar gf-locator:/data
docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "deploy --jar=/data/inference-function-0.0.1-SNAPSHOT.jar"

# Start client app.
# This app loads the inference model and product review seed data.
# It also exposes the APIs to exercise client-side inference and to invoke the function for server-side inference.
cd applications/inference-app && java -jar target/inference-app-0.0.1-SNAPSHOT.jar
