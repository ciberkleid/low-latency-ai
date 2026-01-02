# Low Latency AI

## Option A: Model distribution (edge inference)

### Description
- Standalone Spring application runs inference logic in its JVM and hosts the inference model
- The model has no external data dependency
- GemFire coordinates model distribution, caching, and observability

### Flow
- **Spring app → GemFire:** sends inference inputs
- **GemFire:** checks cache
- **Cache miss**
    - **GemFire → Spring app:** pushes model
    - **Spring app:** runs inference using the model colocated in the JVM
    - **GemFire:** caches and observes inputs/output
- **Cache hit**
    - **GemFire → Spring app:** returns cached inference result

### Benefits
- Centralized model updates
- Low-latency, in-JVM inference
- Inference caching avoids repeated execution
- Observability of inputs and outputs via GemFire

## Option B: Model hosting (data-local inference)

### Description
- A lightweight Spring application runs outside GemFire
- A GemFire function runs the data-intensive Java inference logic
- GemFire hosts the inference model and related data
- Best for data-dependent, data-intensive models

### Flow
- **Spring app → GemFire:** invokes function with inference inputs
- **GemFire:** checks cache
- **Cache miss**
    - **GemFire Function:** runs inference using model and data colocated in GemFire
    - **GemFire:** caches and observes inputs/output
- **Cache hit**
    - **GemFire → Spring app:** returns cached inference result

### Benefits
- Data-local inference avoids data movement and reduces latency
- Distributed, parallel execution (model and data are distributed across GemFire servers)
- Inference caching avoids redundant computation
- Observability of inputs and outputs via GemFire

## Download model

```shell
./deployments/local/model/download-model.sh
```

## Run tests

Execute [integration tests](src/test/java/com/example/low_latency_ai/service/InferenceServiceTest.java)
These tests verify Spring app hosting model, not yet interaction with GemFire.