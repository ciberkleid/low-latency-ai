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
- **loader module**: one-time utility that uploads the ONNX model and tokenizer to GemFire
- **engine module**: runtime component used by application instances to download the model from GemFire into memory, serve inference, and cache results in the `SentimentResults` region

### Model Update Propagation Flow

GemFire Regions are comparable to database tables: they act as the system of record for models and tokenizers, and—like database triggers—can emit change events when those records are updated.
 The Spring application uses these change events to refresh the in-memory model.

#### Initial Model Load (Startup)

- **Loader module → GemFire:** a one-time or administrative process uploads the ONNX model and tokenizer to the `Models` region
- **Engine module → GemFire:** on startup, each engine instance uses a Spring Data GemFire repository to fetch the model by name (for example, key `Sentiment` from the `Models` region)
- **GemFire:** returns the current model and tokenizer stored in the region
- **Engine module:** loads the model into memory and initializes the inference service
- **Engine module:** performs this initial model load once per application instance

#### Model Update Propagation (Runtime)

- **Loader module / operator → GemFire:** uploads an updated model or tokenizer to the `Models` region
- **GemFire:** updates the region entry and emits a change event, analogous to a database trigger firing on update
- **Engine module:** receives the region change event via a listener or trigger
- **Engine module:** responds to the event by reloading the updated model from GemFire
- **Engine module:** replaces or refreshes the in-memory model used for inference
- **Subsequent inference requests:** are served using the updated model

#### Notes and Constraints

- GemFire propagates model update notifications through region change events
- The engine module is responsible for deciding when and how to reload the model in response to change events. In this demo, the engine service supports both constructor-based model initialization (startup) and runtime model replacement (updates).
- This explicit reload mechanism avoids hidden side effects and keeps model lifecycle management under application control
DETE THIS

### Model Update Propagation Flow

This flow describes how inference models are initially loaded at startup and how subsequent model updates are propagated to running application instances.

GemFire Regions are comparable to database tables: they act as the system of record for models and tokenizers and—like database triggers—emit change events when those records are updated. Spring applications consume these events to explicitly manage in-memory model lifecycles.

#### Initial Model Load (Startup)

- **Loader module → GemFire:** uploads the initial version of the ONNX model and tokenizer to the `Models` region
- **Engine module → GemFire:** on startup, each engine instance uses a Spring Data GemFire repository to fetch the model by name (for example, key `Sentiment` from the `Models` region)
- **GemFire:** returns the current model and tokenizer stored in the region
- **Engine module:** loads the model into memory and initializes the inference service

#### Model Update Propagation (Runtime)

- **Loader module:** polls a local directory for changes to the model or tokenizer files
- **Loader module → GemFire:** when a file change is detected, uploads the updated model or tokenizer to the `Models` region
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
- Evaluate sentiment per comment (using the same ONNX model uploaded by the Spring app **loader**)
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

### Download the Model

```shell
./ops/local/model/download-model.sh
```

---

### Start GemFire

Start GemFire using Docker:

```shell
./ops/local/docker/start-gemfire.sh
```

---

### Start the Application

In applications/inference-app:

```shell
mvn spring-boot:run
```

---

## Verifying the Demo

### Verify Model Loaded into GemFire

Run `gfsh` inside the locator container:

```shell
docker exec -it gf-locator gfsh
```

Connect to the cluster:

```gfsh
connect --locator=127.0.0.1[10334]
list regions
query --query="select * from /AiModel.keys"
```

---

### Verify Model Loaded into the Application

Look for the following log entry:

```txt
Executing onnx inference service using text: This is just a text for started to initialize the loader. This will failed is the loader is not loaded in GemFire
```

---

### Verify Inference Caching

Query cached inference results:

```gfsh
query --query="select key,value from /SentimentResults.entries"
```

Or invoke the sentiment endpoint:

```shell
open http://localhost:8080
```

On the first request, logs will show inference execution and a cache `PutOp`.
Subsequent requests with the same input will show only a cache `GetOp`.

Example log output:

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
Within a few seconds, you should see:

```txt
2026-01-16T11:51:33.380-05:00  INFO 96553 --- [low-latency-ai] [   scheduling-1] c.e.l.loader.AiModelResourceMonitor      : Detected change in AI assets (model: false, tokenizer: true). Refreshing GemFire entry.
2026-01-16T11:51:33.542-05:00 DEBUG 96553 --- [low-latency-ai] [   scheduling-1] o.a.g.cache.client.internal.AbstractOp   : PutOpImpl constructing message for EventID[id=39 bytes;threadID=283533;sequenceID=0]; operation=UPDATE
2026-01-16T11:51:38.416-05:00  INFO 96553 --- [low-latency-ai] [   scheduling-1] c.e.l.e.service.OnnxInferenceService     : Update event detected: Received new model from GemFire
2026-01-16T11:51:38.416-05:00  INFO 96553 --- [low-latency-ai] [   scheduling-1] c.e.l.e.service.OnnxInferenceService     : Updating local Onnx session and tokenizer with new model.
```

This confirms that local model changes propagate through GemFire and update in-memory inference sessions.

---

## Tests

Run integration tests:

```shell
./mvnw test
```

---

## TODO

1. Invalidate locally cached models when the model in GemFire is updated
2. Invalidate or version inference cache entries on model changes
3. Expand data-local inference test coverage
