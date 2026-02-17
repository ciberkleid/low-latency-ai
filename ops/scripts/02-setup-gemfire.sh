# Bootstraps a local Apache Geode/GemFire environment using Docker.
# Script flow:
# 1) Creates a dedicated Docker bridge network.
# 2) Starts a locator and waits for JMX readiness.
# 3) Applies cluster PDX serialization configuration.
# 4) Starts a cache server connected to the locator.
# 5) Creates application regions used by this project.
# Intended use: local development and manual environment setup.

# 1) Create an isolated Docker network for GemFire containers.
docker network create gemfire --driver bridge

# 2) Start the GemFire locator with JMX, management, and metrics ports exposed.
docker run -d -e 'ACCEPT_TERMS=y' --rm --name gf-locator --network=gemfire \
  -p 10334:10334 -p 1099:1099 -p 7070:7070 -p 7999:7999 -p 7777:7777 \
  gemfire/gemfire:10.2-jdk21 \
  gfsh start locator --name=locator1 \
  --jmx-manager-hostname-for-clients=gf-locator \
  --hostname-for-clients=gf-locator \
  --J=-Dgemfire.prometheus.metrics.emission=Default \
  --J=-Dgemfire.prometheus.metrics.port=7777  \
  --J=-Duser.timezone=America/New_York \
  --J=-Dgemfire.prometheus.metrics.interval=15s \
  --J=-Djava.rmi.server.hostname=gf-locator \
  --J=-Dgemfire.tcp-port=7999

# Wait for locator readiness until JMX connection succeeds.
until docker exec -it  gf-locator  gfsh -e "connect --jmx-manager=gf-locator[1099]" >/dev/null 2>&1; do
  echo "Waiting for locator to start..."
  sleep 2
done
echo "Locator is up"

# 3) Apply cluster-wide PDX serialization settings through the locator.
# Notes: Configure PDX to deserialize to Java objects on reads (read-serialized=false) rather than to read in serialized PDX format,
#        with hint about the classes to use for deserialization (not needed for client app as Spring Data Gemfire provides the hints),
#        and persist PDX metadata in default disk store (persistence of data itself is configured at the Region level.
docker exec -it gf-locator gfsh \
  -e "connect --jmx-manager=gf-locator[1099]" \
  -e "configure pdx --read-serialized=false --auto-serializable-classes=com.example.low_latency_ai.domain.AiModel,com.example.low_latency_ai.domain.ProductReview --disk-store"
# -e "configure pdx --read-serialized=true --disk-store"

# 4) Start a cache server member connected to the locator.
docker run -d -e 'ACCEPT_TERMS=y' --rm --name gf-server1 --network=gemfire \
  -p 40404:40404 -p 7080:7080 -p 7977:7977 \
  gemfire/gemfire:10.2-jdk21 \
  gfsh start server --name=server1 --locators=gf-locator\[10334\] \
  --hostname-for-clients=gf-server1 \
  --start-rest-api=true \
  --http-service-port=7080 \
  --J=-Dgemfire.prometheus.metrics.emission=Default \
  --J=-Dgemfire.prometheus.metrics.port=7977  \
  --J=-Duser.timezone=America/New_York \
  --J=-Dgemfire.prometheus.metrics.interval=15s

# Give the server a moment to initialize before region creation.
sleep 5

# 5) Create the regions for the primary model, sentiment output, and product reviews.
# Defaults: redundancy is 0 and data persistence is disabled. Statistics are required for expiration/metrics.
REGION_NAMES=(AiModel SentimentResults ProductReviews)
for REGION_NAME in "${REGION_NAMES[@]}"; do
  docker exec -it gf-locator gfsh \
    -e "connect --jmx-manager=gf-locator[1099]" \
    -e "create region --name=$REGION_NAME --type=PARTITION  --enable-statistics=true"
done

echo "GemFire setup complete. To connect with gfsh, run:"
echo "docker exec -it gf-locator gfsh"
echo "Then run 'connect'"