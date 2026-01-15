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
./deployments/local/loader/download-loader.sh
```

## Start GemFire

GemFire Regions are comparable to DB tables
Can use a trigger on a Region to pull changes to an uploaded AI Model
GemFire will ensure that the app has the latest version of the model when the app starts up:
- App starts and uses Spring Data Gemfire repository to get the model by name (e.g. SentimentModel) from a Models Region within Gemfire with Key=Sentiment and Value=<the value of the tokenizer of the Model>.
- App reads this one time
- Need a signal to repull or push model when the model is modified
- Service needs a way to provide model in constructor and also a way to update the model


Start Gemfire on Docker

```shell
./deployments/local/docker/start.sh
```

## Start Application

Start the application

```shell
./mvnw spring-boot:run
```

To verify that the loader module loaded the model into GemFire, you can run the `gfsh` CLI from within the `gf-locater` Docker container.

```shell
docker exec -it gf-locator gfsh
```

Connect to cluster where 10334 is the locator port

```gfsh
connect --locator=127.0.0.1[10334]
list regions
query --query="select * from /AiModel.keys"
```

To verify that the engine module has pulled the model from GemFire, look for the following line in the application log file:
```txt
Executing onnx inference service using text: This is just a text for started to initialize the loader. This will failed is the loader is not loaded in GemFire
```

To verify that the engine cached the prompt and inference result in GemFire, run:
```gfsh
query --query="select key,value from /SentimentResults.entries"
```

Or you can send any string to the model to evaluate the sentiment:
```shell
open http://localhost:8080  # Find and execute the checkSentiment endpoint
```

To verify that an initial prompt uses the model and subsequent requests with the same prompt return the cached result, check the log file. It will look something like this, showing an inference operation and a cache PutOp the first time, and only a cache GetOp on subsequent requests:
```txt
2026-01-15T10:38:54.959-05:00 DEBUG 42167 --- [low-latency-ai] [nio-8080-exec-5] o.a.g.cache.client.internal.AbstractOp   : constructing a GetOp for key "Spring is awesome"
2026-01-15T10:38:54.966-05:00  INFO 42167 --- [low-latency-ai] [nio-8080-exec-5] c.e.l.e.service.OnnxInferenceService     : Executing onnx inference service using text: "Spring is awesome"
2026-01-15T10:38:54.983-05:00  INFO 42167 --- [low-latency-ai] [nio-8080-exec-5] c.e.l.e.service.OnnxInferenceService     : Sentiments results: POSITIVE
2026-01-15T10:38:54.983-05:00 DEBUG 42167 --- [low-latency-ai] [nio-8080-exec-5] o.a.g.cache.client.internal.AbstractOp   : PutOpImpl constructing message for EventID[id=39 bytes;threadID=434440;sequenceID=0]; operation=UPDATE
2026-01-15T10:38:55.778-05:00 DEBUG 42167 --- [low-latency-ai] [nio-8080-exec-6] o.a.g.cache.client.internal.AbstractOp   : constructing a GetOp for key "Spring is awesome"
```


## Run tests

Execute [integration tests](src/test/java/com/example/low_latency_ai/service/InferenceServiceTest.java)
These tests verify Spring app hosting model, not yet interaction with GemFire.
