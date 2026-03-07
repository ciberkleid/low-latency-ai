#!/usr/bin/env bash
set -euo pipefail

# Model: distilbert-base-uncased-finetuned-sst-2-english
#
# A lightweight transformer model for binary sentiment analysis in English.
#
# - DistilBERT: a smaller, faster version of BERT created via knowledge distillation
# - Base: standard DistilBERT model size (6 transformer layers)
# - Uncased: input text is lowercased before processing
# - Fine-tuned on SST-2: trained specifically to classify sentiment
# - English-only: trained and evaluated on English text
#
# Note:
# This model outputs two logits: [negative, positive].
# It always predicts positive or negative (SST-2 has no neutral class).

DEST=models/distilbert/distilbert-base-uncased-finetuned-sst-2-english

echo "==> Checking model artifacts..."
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

echo "==> Checking Java version..."
# Require Java 21 for compatibility with GemFire.
JAVA_MAJOR="$(java -version 2>&1 | awk -F '[\".]' '/version/ {print $2; exit}')"
if [[ "$JAVA_MAJOR" != "21" ]]; then
  echo "Java 21 is required for compatibility with GemFire. Current Java major version: ${JAVA_MAJOR:-unknown}." >&2
  echo "Set Java to 21 and re-run. Example: sdk install java 21.0.10-librca && sdk use java 21.0.10-librca" >&2
  exit 1
fi
echo "    Java $JAVA_MAJOR detected."

# Build all Java modules so app and function artifacts are available:
# - libraries/shared-domain
# - functions/inference-function
# - applications/inference-app
echo "==> Building Java modules..."
./mvnw -DskipTests clean install
echo "    Build complete."

# Start the app; Spring Boot Docker Compose integration manages:
# - GemFire startup from ops/gemfire/docker-compose.yml
# - Region initialization (regions-init profile)
# - Function deployment (function-deploy profile)
echo "==> Starting application..."
cd applications/inference-app && java -jar target/inference-app-0.0.1-SNAPSHOT.jar
