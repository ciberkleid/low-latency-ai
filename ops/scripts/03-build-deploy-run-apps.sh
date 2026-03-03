#!/usr/bin/env bash
set -euo pipefail

# Require Java 21 for compatibility with GemFire.
JAVA_MAJOR="$(java -version 2>&1 | awk -F '[\".]' '/version/ {print $2; exit}')"
if [[ "$JAVA_MAJOR" != "21" ]]; then
  echo "Java 21 is required for compatibility with GemFire. Current Java major version: ${JAVA_MAJOR:-unknown}." >&2
  echo "Set Java to 21 and re-run. Example: sdk install java 21.0.10-librca && sdk use java 21.0.10-librca" >&2
  exit 1
fi

# Build all Java modules so app and function artifacts are available.
./mvnw -DskipTests clean install

# Start the app; Spring Boot Docker Compose integration manages:
# - GemFire startup from ops/gemfire/docker-compose.yml
# - Region initialization (regions-init profile)
# - Function deployment (function-deploy profile)

cd applications/inference-app && java -jar target/inference-app-0.0.1-SNAPSHOT.jar
