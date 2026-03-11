#!/usr/bin/env bash
set -euo pipefail

# Startup script for the low-latency AI inference demo.
#
# Steps:
#   1. Download model artifacts (skipped if already present)
#      Model: distilbert-base-uncased-finetuned-sst-2-english — a lightweight
#        transformer for binary sentiment analysis (positive/negative only, no neutral).
#   2. Verify Java 21 is active (required for GemFire compatibility)
#   3. Build all Java modules (shared-domain, inference-function, inference-app)
#   4. Start inference-app, which triggers Spring Boot Docker Compose to:
#        - Start GemFire in Docker (ops/gemfire/docker-compose.yml)
#        - Create regions (regions-init profile)
#        - Deploy inference-function (function-deploy profile)
#      NOTE: If Spring Boot Docker Compose is disabled in inference-app
#            (spring.docker.compose.enabled=false), run optional-setup-gemfire.sh
#            to start GemFire and initialize regions manually before this script.
#
# End state:
#   - GemFire cluster running with regions, inference-function, model, and
#     product reviews data all loaded
#  - inference-app running on port 8080 with model loaded into memory


# Step 1: Download model artifacts (skipped if already present)
DEST=models/distilbert/distilbert-base-uncased-finetuned-sst-2-english
echo -e "\n==> [1/4] Checking model artifacts...\n"
if [[ -f "$DEST/model.onnx" && -f "$DEST/tokenizer.json" ]]; then
  echo "    Model files already present in $DEST, skipping download."
else
  echo "    Downloading model artifacts to $DEST..."
  mkdir -p "$DEST"
  cd "$DEST"
  wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx
  wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/tokenizer.json
  cd -
  echo "    Model artifacts downloaded."
fi

# Step 2: Verify Java 21 is active (required for GemFire compatibility)
echo -e "\n==> [2/4] Checking Java version...\n"
JAVA_MAJOR="$(java -version 2>&1 | awk -F '[\".]' '/version/ {print $2; exit}')"
if [[ "$JAVA_MAJOR" != "21" ]]; then
  echo "Java 21 is required for compatibility with GemFire. Current Java major version: ${JAVA_MAJOR:-unknown}." >&2
  echo "Set Java to 21 and re-run. Example: sdk install java 21.0.10-librca && sdk use java 21.0.10-librca" >&2
  exit 1
fi
echo "    Java $JAVA_MAJOR detected."

# Step 3: Build all Java modules (shared-domain, inference-function, inference-app)
echo -e "\n==> [3/4] Building Java modules...\n"
./mvnw -DskipTests clean install
echo "    Build complete."

# Step 4: Start inference-app; Spring Boot Docker Compose integration manages:
#   - GemFire startup (ops/gemfire/docker-compose.yml)
#   - Region initialization (regions-init profile)
#   - Function deployment (function-deploy profile)
#   On startup, the app also:
#   - Loads the model into GemFire
#   - Loads product reviews data into GemFire (ops/data/product-reviews.csv)
#   - Submits a warm-up sentiment request, which triggers the model to be pulled
#     from GemFire into local memory; the result is cached in GemFire
echo -e "\n==> [4/4] Starting application...\n"
cd applications/inference-app && java -jar target/inference-app-0.0.1-SNAPSHOT.jar
