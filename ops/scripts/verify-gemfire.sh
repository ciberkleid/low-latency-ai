#!/usr/bin/env bash
set -euo pipefail

# GemFire verification helper.
#
# Use this script to quickly run common checks instead of manually searching
# through a long list of gfsh examples.
#
# Examples:
#   ./ops/scripts/05-verify-gemfire.sh status
#   ./ops/scripts/05-verify-gemfire.sh regions
#   ./ops/scripts/05-verify-gemfire.sh functions
#   ./ops/scripts/05-verify-gemfire.sh sentiments
#   ./ops/scripts/05-verify-gemfire.sh reviews
#   ./ops/scripts/05-verify-gemfire.sh product "Lawnmower"
#   ./ops/scripts/05-verify-gemfire.sh exec-fn "Coffee Mug"

LOCATOR_CONTAINER="${LOCATOR_CONTAINER:-gf-locator}"

usage() {
  cat <<'EOF'
Usage:
  05-verify-gemfire.sh <check>

Checks:
  status       Show compose/container status.
  regions      List GemFire regions.
  functions    List deployed functions.
  metrics      Show cluster metrics and SentimentResults metrics.
  sentiments   Show cached sentiment results.
  reviews      Show ProductReviews entries.
  product NAME Show grouped review counts for one product.
  exec-fn NAME Execute countPositiveReviews for one product.
  logs         Show recent server1 log lines.
  baseline     Run core post-start checks (AiModel + sentiments + reviews).
  help         Show this help.

Workflow:
  1) After startup:         baseline
  2) After edge requests:   sentiments
  3) After function calls:  product "<name>", then sentiments
EOF
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker command not found" >&2
    exit 1
  fi
}

run_gfsh() {
  local cmd="$1"
  docker exec -i "${LOCATOR_CONTAINER}" gfsh \
    -e "connect --jmx-manager=gf-locator[1099]" \
    -e "${cmd}"
}

status() {
  docker compose -f ops/gemfire/docker-compose.yml ps
}

regions() {
  run_gfsh "list regions"
}

functions() {
  run_gfsh "list functions"
}

metrics() {
  run_gfsh "show metrics"
  run_gfsh "show metrics --region=/SentimentResults"
}

sentiments() {
  run_gfsh "query --query='select key,value from /SentimentResults.entries'"
}

reviews() {
  run_gfsh "query --query='select key,value.productName,value.review from /ProductReviews.entries'"
}

product_counts() {
  local product_name="$1"
  run_gfsh "query --query=\"select value.review, count(*) as reviewCount from /ProductReviews.entries where value.productName='${product_name}' group by value.review\""
}

exec_fn() {
  local product_name="$1"
  run_gfsh "execute function --id=countPositiveReviews --region=/ProductReviews --arguments='${product_name}'"
}

logs() {
  run_gfsh "show log --member=server1 --lines=100"
}

baseline() {
  run_gfsh "query --query='select * from /AiModel.keys'"
  sentiments
  reviews
}

main() {
  require_docker

  local check="${1:-help}"
  case "${check}" in
    status) status ;;
    regions) regions ;;
    functions) functions ;;
    metrics) metrics ;;
    sentiments) sentiments ;;
    reviews) reviews ;;
    product)
      shift
      if [[ $# -lt 1 ]]; then
        echo "Missing product name. Example: product \"Lawnmower\"" >&2
        exit 1
      fi
      product_counts "$1"
      ;;
    exec-fn)
      shift
      if [[ $# -lt 1 ]]; then
        echo "Missing product name. Example: exec-fn \"Coffee Mug\"" >&2
        exit 1
      fi
      exec_fn "$1"
      ;;
    logs) logs ;;
    baseline) baseline ;;
    help|-h|--help) usage ;;
    *)
      echo "Unknown check: ${check}" >&2
      echo
      usage
      exit 1
      ;;
  esac
}

main "$@"
