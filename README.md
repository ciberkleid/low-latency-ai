# Low Latency AI Inference

This repository demonstrates low-latency AI inference with GemFire using two patterns: 
- **Edge inference:** models distributed by GemFire to standalone applications for local execution, suited to inference without external data dependencies
- **Data-local inference:** models, data, and execution colocated in GemFire, suited to inference that depends on stored data

GemFire provides model storage/distribution, including region events to propagate model updates to running application instances. In the data-local pattern, GemFire provides region-based function execution so inference runs where the data resides. Finally, across both patterns, GemFire also serves as a shared inference-result cache to reduce repeated computation and improve response time.

The example model is a sentiment classifier in [ONNX](https://onnx.ai), an open, framework-neutral model format. In this project, inference is executed with [ONNX Runtime](https://onnxruntime.ai/) in both the GemFire function (data-local path) and the standalone application (edge path), allowing one exported model artifact to be reused across both; while other formats are possible, they typically introduce more framework-specific integration overhead in this JVM-based setup.

---

## Edge Inference (Model Distribution)

### Use Case
Evaluate sentiment for a single user-submitted string:

```txt
"I love Spring" → POSITIVE
```

### Description
- A standalone Spring application pulls the inference model from GemFire and hosts it in memory, running inference logic in its JVM
- The inference model has no external data dependencies
- GemFire stores and distributes the model and coordinates caching and observability


### Flow
- **Spring app → GemFire:** submits inference input
- **GemFire:** checks the inference cache

**Cache miss**
- **GemFire → Spring app:** provides the model (if not already loaded)
- **Spring app:** runs inference in-JVM using the colocated model
- **GemFire:** caches and observes inputs and outputs

**Cache hit**
- **GemFire → Spring app:** returns the cached inference result

### Components
- **model module**: one-time utility that uploads the ONNX model and tokenizer to GemFire
- **engine module**: runtime component used by application instances to download the model from GemFire into memory, serve inference, and cache results in the `SentimentResults` region

### Model Update Propagation Flow

This flow describes how inference models are initially loaded at startup and how subsequent model updates are propagated to running application instances.

GemFire Regions are comparable to database tables: they act as the system of record for models and tokenizers and—like database triggers—emit change events when those records are updated. Spring applications consume these events to explicitly manage in-memory model lifecycles.

#### Initial Model Load (Startup)

- **Model module → GemFire:** uploads the initial version of the ONNX model and tokenizer to the `Models` region
- **Engine module → GemFire:** on startup, each engine instance uses a Spring Data GemFire repository to fetch the model by name (for example, key `Sentiment` from the `Models` region)
- **GemFire:** returns the current model and tokenizer stored in the region
- **Engine module:** loads the model into memory and initializes the inference service

#### Model Update Propagation (Runtime)

- **Model module:** polls a local directory for changes to the model or tokenizer files
- **Model module → GemFire:** when a file change is detected, uploads the updated model or tokenizer to the `Models` region
- **GemFire:** updates the region entry and emits a region change event
- **Engine module:** receives the region event indicating that the model has changed
- **Engine module:** explicitly reloads the updated model from GemFire
- **Engine module:** replaces or refreshes the in-memory model used for inference
- **Subsequent inference requests:** use the updated model

### Benefits
- Centralized model distribution and updates
- Low-latency, in-process inference
- Shared inference cache across application instances
- Observability of inference inputs and outputs via GemFire

---

## Data-Local Inference (Model, Data, and Execution Colocation)

### Use Case
- Submit a product ID
- Query the `ProductReviews` region for comments
- Evaluate sentiment per comment (using the same ONNX model uploaded by the Spring app ** model** module)
- Return the percentage of positive reviews
- Cache per-comment results in the shared `SentimentResults` region

### Description
- A lightweight Spring application runs outside GemFire
- A GemFire function runs the data-dependent Java inference logic
- GemFire hosts the inference model and the related data
- Best suited for data-dependent or data-intensive inference

### Flow
- **Spring app → GemFire:** invokes a function with inference inputs
- **GemFire:** checks the inference cache

**Cache miss**
- **GemFire function:** runs inference using model and data colocated in GemFire
- **GemFire:** caches and observes inputs and outputs

**Cache hit**
- **GemFire → Spring app:** returns the cached inference result

### Benefits
- Data-local execution avoids data movement
- Distributed, parallel inference (model and data are distributed across GemFire servers)
- Shared inference cache avoids redundant computation
- Observability of inference inputs and outputs via GemFire

---

## Inference Caching (Shared Capability)

Both inference architectures use GemFire as a shared, low-latency cache for inference inputs and outputs.

- Cache hits avoid repeated inference execution
- Cached results are shared across application instances and inference modes
- Inputs and outputs are observable through GemFire tooling

---

## Running the Demo

### Pre-requisite
You may need to add the locator and member to your `/etc/hosts` file, as shown here:
```shell
~ $ cat /etc/hosts
##
# Host Database
#
# localhost is used to configure the loopback interface
# when the system is booting.  Do not change this entry.
##
127.0.0.1	localhost
127.0.0.1       gf-locator
127.0.0.1       gf-server1
```

---
### Start Everything

This single script starts the full demo environment:
- Downloads the model artifacts (skipped if already present)
- Builds all Java modules (`shared-domain`, `inference-function`, `inference-app`)
- Starts `inference-app`, which triggers Spring Boot Docker Compose to:
  - Start GemFire in Docker
  - Create the required regions (`AiModel`, `ProductReviews`, `SentimentResults`)
  - Deploy `inference-function` to GemFire

```shell
./ops/scripts/_startup.sh
```

The client application will be running in the terminal window where you ran the script.

> **Note:** If Spring Boot Docker Compose support is disabled, use `./ops/scripts/optional-setup-gemfire.sh` to manually start GemFire and initialize the regions before running the application.

---
### Verify Startup

**Model files**

Confirm the model artifacts are present locally:
```shell
$ tree models
models
└── distilbert
    └── distilbert-base-uncased-finetuned-sst-2-english
        ├── model.onnx
        └── tokenizer.json
```

**Application startup**

Confirm the app loaded the model from GemFire by looking for this log entry:
```txt
Executing onnx inference service using text: Woohoo! This entry ensures client engine module pulls model from GemFire at startup. Well done!
```

**GemFire**

Confirm the two GemFire Docker containers are running:
```shell
$ docker ps --format 'table {{.ID}}\t{{.Names}}'
CONTAINER ID   NAMES
f97a8af4b73b   gf-server1
fc02f25ecc19   gf-locator
```

To run commands in GemFire, you can use `docker exec` or the interactive `gfsh` CLI.

Option 1 — `docker exec` (non-interactive, good for quick checks):
```shell
$ docker exec -it gf-locator gfsh \
    -e "connect --jmx-manager=gf-locator[1099]" \
    -e "list regions" \
    -e "list functions"
```

Expected output:
```
List of regions
----------------
AiModel
ProductReviews
SentimentResults

Member  | Function
------- | --------------------
server1 | countPositiveReviews
```

Option 2 — interactive `gfsh` session:
```shell
docker exec -it gf-locator gfsh
```

Then at the `gfsh` prompt:
```gfsh
connect
list regions
list functions
```

To shut down the cluster and stop the Docker containers:
```gfsh
shutdown --include-locators
```

For additional queries including cached inference results in `SentimentResults`, run:
```shell
./ops/scripts/verify-gemfire.sh
```

---
### Send Edge Requests

Send sample client-side inference requests.
This script exercises the client app API that executes inference locally with the model in-memory.

```shell
./ops/scripts/send-edge-requests.sh
```


The Sentiment region on server1 is used to cache inference results.
On the first request, logs will show inference execution and a cache `PutOp`.
Subsequent requests with the same input will show only a cache `GetOp`.
The script output will show the latency of the request.
Notice also the substantial improvement in response time with a cache hit after the first execution (an order of magnitude).

You can also check the client log output. It should look something like this:

```txt
2026-01-15T10:38:54.959-05:00 DEBUG 42167 --- [low-latency-ai] [nio-8080-exec-5] o.a.g.cache.client.internal.AbstractOp   : constructing a GetOp for key "Spring is awesome"
2026-01-15T10:38:54.966-05:00  INFO 42167 --- [low-latency-ai] [nio-8080-exec-5] c.e.l.e.service.OnnxInferenceService     : Executing onnx inference service using text: "Spring is awesome"
2026-01-15T10:38:54.983-05:00  INFO 42167 --- [low-latency-ai] [nio-8080-exec-5] c.e.l.e.service.OnnxInferenceService     : Sentiments results: POSITIVE
2026-01-15T10:38:54.983-05:00 DEBUG 42167 --- [low-latency-ai] [nio-8080-exec-5] o.a.g.cache.client.internal.AbstractOp   : PutOpImpl constructing message for EventID[id=39 bytes;threadID=434440;sequenceID=0]; operation=UPDATE
2026-01-15T10:38:55.778-05:00 DEBUG 42167 --- [low-latency-ai] [nio-8080-exec-6] o.a.g.cache.client.internal.AbstractOp   : constructing a GetOp for key "Spring is awesome"
```

---

### Verify Model Updates


Modify the local model or tokenizer (adding an empty line to `tokenizer.json` is sufficient).
You can also use this script to make the change:

```shell
./ops/scripts/update-model.sh
```

Within a few seconds, you should see:

```txt
2026-02-18T01:36:22.744-05:00  INFO 29850 --- [low-latency-ai] [   scheduling-1] c.e.l.model.AiModelResourceMonitor       : Detected change in AI assets (model: false, tokenizer: true). Refreshing GemFire entry.
2026-02-18T01:36:23.086-05:00 DEBUG 29850 --- [low-latency-ai] [   scheduling-1] o.a.g.cache.client.internal.AbstractOp   : PutOpImpl constructing message for EventID[id=39 bytes;threadID=823656;sequenceID=1]; operation=UPDATE
2026-02-18T01:36:25.891-05:00  INFO 29850 --- [low-latency-ai] [   scheduling-1] c.e.l.e.service.OnnxInferenceService     : Update event detected: Received new model from GemFire.
2026-02-18T01:36:25.891-05:00  INFO 29850 --- [low-latency-ai] [   scheduling-1] c.e.l.e.service.OnnxInferenceService     : Updating local Onnx session and tokenizer with new model.
```

This confirms that local model changes propagate through GemFire and update in-memory inference sessions without any application downtime.

---

### Send Function Requests

Next, exercise the server-side function endpoint:

Here too, the script output will show the latency of the request.
Notice again the substantial improvement in response time with a cache hit after the first execution.

```shell
./ops/scripts/send-function-requests.sh
```

You can also reference `verify-gemfire.sh` again for additional queries to inspect changes to the SentimentResults region or to view the server-side logging showing the execution of the function (hint: run `show log --member=server1 --lines=100`).

---

## Tests

Run integration tests:

```shell
./mvnw test
```

## Acknowledgments

Special thanks to the following contributors whose collaboration and expertise made this project possible:

- [Siva Edaichamy](https://github.com/siva-edaichamy) – Original concept and direction
- [Gregory Green](https://github.com/ggreen) – Primary implementation
- [Udo Kohlmeyer](https://github.com/kohlmu-pivotal) – Critical debugging support

---

## TODO

1. Update model in function when model in AiModel Region is updated.
2. Invalidate or version inference cache entries on model changes
3. Expand data-local inference test coverage
4. Avoid adding an entry to etc hosts file
