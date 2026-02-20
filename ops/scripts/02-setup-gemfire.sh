#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a local Apache Geode/GemFire environment using Docker.
# Intended use: local development and manual environment setup.

# Compose flow in docker-compose.yml:
# 1) Start gf-locator and wait for healthcheck readiness.
# 2) Run gf-pdx-config to apply PDX cluster configuration.
# 3) Start gf-server1 after locator and PDX config are complete.
# 4) Run gf-regions-init to create required regions.

# Compose lifecycle commands quick reference:
#   up -d   : create/start services in the background.
#   stop    : stop services but keep containers/network for later reuse.
#   start   : start previously stopped services without recreating.
#   down    : stop and remove services and network (and container-local state).
#             if named volumes are enabled for persistence (see compose file), use `docker compose down -v` to wipe persisted data.
# Environment variables:
#   COMPOSE_FILE     : compose file path. If unset, defaults to ops/gemfire/docker-compose.yml.

# Set or validate compose file
# docker compose will take file name from COMPOSE_FILE if set
if [[ -z "${COMPOSE_FILE:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  COMPOSE_FILE="${SCRIPT_DIR}/../gemfire/docker-compose.yml"
  export COMPOSE_FILE
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Compose file not found: ${COMPOSE_FILE}" >&2
  ls -la "$(dirname "${COMPOSE_FILE}")" >&2 || true
  exit 1
fi

# Start locator + server:
# File set via $COMPOSE_FILE; run in detached mode (-d)
docker compose up -d

echo
for c in gf-pdx-config gf-regions-init; do
    code=$(docker inspect "$c" --format '{{.State.ExitCode}}')
    echo "$c: $([ "$code" -eq 0 ] && echo OK || echo FAIL) (exit=$code)"
  done

# Verify:
# docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "list regions"

# To check status:
# docker compose ps

# To start gfsh:
# `docker compose exec gf-locator gfsh` OR `docker exec -it gf-locator gfsh`

echo
cat <<'EOF'
GemFire setup complete.

The following commands are quick operational references for day-to-day local use:

Compose lifecycle:
  docker compose up -d     (already run by this script)
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
