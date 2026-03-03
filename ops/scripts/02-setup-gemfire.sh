#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# MANUAL GEMFIRE SETUP (FALLBACK PATH)
# Use this script only when Spring Boot Docker Compose integration is disabled
# for the app, or when you intentionally manage GemFire separately.
# If Spring Boot Docker Compose is enabled, start the app directly instead of
# running this script.
# ============================================================================
#
# Bootstraps a local Apache Geode/GemFire environment using Docker Compose.

# Compose flow in docker-compose.yml:
# 1) Start gf-locator and wait for healthcheck readiness.
# 2) Run gf-pdx-config to apply PDX cluster configuration.
# 3) Start gf-server1 after locator and PDX config are complete.
# 4) Run gf-regions-init to create required regions.
# 5) Optionally run gf-function-deploy to deploy the server-side function.

# Compose lifecycle commands quick reference:
#   up -d   : create/start services in the background.
#   stop    : stop services but keep containers/network for later reuse.
#   start   : start previously stopped services without recreating.
#   down    : stop and remove services and network (and container-local state).
#             if named volumes are enabled for persistence (see compose file), use `docker compose down -v` to wipe persisted data.
# Environment variables:
#   COMPOSE_FILE     : compose file path. If unset, defaults to ops/gemfire/docker-compose.yml.
#   COMPOSE_PROFILES : comma-separated profiles to start.
#                      Defaults to regions-init,function-deploy.

# Set or validate compose file
# docker compose will take file name from COMPOSE_FILE if set
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${COMPOSE_FILE:-}" ]]; then
  COMPOSE_FILE="${SCRIPT_DIR}/../gemfire/docker-compose.yml"
  export COMPOSE_FILE
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Compose file not found: ${COMPOSE_FILE}" >&2
  ls -la "$(dirname "${COMPOSE_FILE}")" >&2 || true
  exit 1
fi

# Default profiles match application.properties behavior when Spring Boot manages compose.
COMPOSE_PROFILES="${COMPOSE_PROFILES:-regions-init,function-deploy}"
PROFILE_ARGS=()
IFS=',' read -r -a profiles <<< "${COMPOSE_PROFILES}"
for profile in "${profiles[@]}"; do
  PROFILE_ARGS+=(--profile "${profile}")
done

# If function-deploy profile is active, ensure function JAR exists before startup.
if [[ ",${COMPOSE_PROFILES}," == *",function-deploy,"* ]]; then
  FUNCTION_JAR="${REPO_ROOT}/functions/inference-function/target/inference-function-0.0.1-SNAPSHOT.jar"
  if [[ ! -f "${FUNCTION_JAR}" ]]; then
    echo "Function JAR not found: ${FUNCTION_JAR}" >&2
    echo "Build it first (example): ./mvnw -DskipTests clean package" >&2
    exit 1
  fi
fi

# Start selected compose profiles, detached, and wait for health/readiness.
docker compose "${PROFILE_ARGS[@]}" up -d --wait

echo
for c in gf-pdx-config gf-regions-init gf-function-deploy; do
    if ! docker container inspect "$c" >/dev/null 2>&1; then
      echo "$c: SKIPPED (container not created; profile may be disabled)"
      continue
    fi
    code=$(docker inspect "$c" --format '{{.State.ExitCode}}')
    echo "$c: $([ "$code" -eq 0 ] && echo OK || echo FAIL) (exit=$code)"
  done

echo
cat <<'EOF'
GemFire setup complete.

Note:
  This is a manual/fallback setup path.
  If Spring Boot Docker Compose integration is enabled, run the app directly
  and let Spring Boot manage these services.

Verification:
  docker compose ps
  docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "list regions"

Operational references:

Compose lifecycle:
  docker compose --profile regions-init --profile function-deploy up -d --wait
                       (already run by this script with default COMPOSE_PROFILES)
  docker compose start     (start previously stopped containers)
  docker compose stop      (stop containers, keep them)
  docker compose down      (stop and remove containers and network)
  docker compose down -v   (also remove named volumes for the project)

Connect with gfsh:
  docker compose exec gf-locator gfsh
  connect --jmx-manager=gf-locator[1099]

Graceful cluster stop (inside gfsh):
  shutdown --include-locators
EOF
