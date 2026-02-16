package com.example.low_latency_ai.gemfire.functions.sentiment;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import com.example.low_latency_ai.model.domain.Sentiment;
import com.example.low_latency_ai.model.domain.AiModel;
import org.apache.geode.cache.Cache;
import org.apache.geode.cache.CacheFactory;
import org.apache.geode.cache.Region;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

public class OnnxPositiveSentimentCounter implements  PositiveSentimentCounter {
    private static final String INPUT_IDS = "input_ids";
    private static final String ATTENTION_MASK = "attention_mask";
    private static final String SENTIMENT_REGION = "SentimentResults";
    private Supplier<AiModel> aiModelSupplier;

    private OrtEnvironment env;
    private OrtSession session;
    private HuggingFaceTokenizer tokenizer;
    private Logger log = LogManager.getLogger(OnnxPositiveSentimentCounter.class);

    public OnnxPositiveSentimentCounter(Supplier<AiModel> aiModelSupplier){
        this.aiModelSupplier = aiModelSupplier;
    }

    @Override
    public Long count(List<String> comments) {
        List<Sentiment> sentiments = execute(comments);
        return sentiments.stream()
                .filter(sentiment -> sentiment == Sentiment.POSITIVE)
                .count();

    }

    private synchronized void ensureInitialized() {
        if (this.tokenizer == null) {
            this.setupSessionAndTokenizer();
        }
    }

    private void setupSessionAndTokenizer() {
        try {
            this.env = OrtEnvironment.getEnvironment();
            var aiModel = this.aiModelSupplier.get();

            this.session = env.createSession(aiModel.getModel(), new OrtSession.SessionOptions());
            this.tokenizer = HuggingFaceTokenizer.newInstance(new ByteArrayInputStream(aiModel.getTokens()), Map.of());
        } catch (OrtException e) {
            throw new UncheckedIOException(new IOException(e));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    protected Region<String, Sentiment> getSentimentRegion() {
        try {
            Cache cache = CacheFactory.getAnyInstance();
            Region<String, Sentiment> region = cache.getRegion(SENTIMENT_REGION);
            if (region == null) {
                log.warn("GemFire cache region {} not found; proceeding without caching", SENTIMENT_REGION);
            }
            return region;
        } catch (RuntimeException e) {
            log.warn("GemFire cache not available for sentiment cache region; proceeding without caching", e);
            return null;
        }
    }

    // Multi-text, throughput-optimized path (batchEncode + one ONNX call)
    public List<Sentiment> execute(List<String> texts) {

        if (texts == null || texts.isEmpty()) {
            log.warn("No texts to process");
            return List.of();
        }
        Region<String, Sentiment> cacheRegion = getSentimentRegion();
        List<String> misses = new ArrayList<>();
        List<Sentiment> results = new ArrayList<>(texts.size());

        for (String text : texts) {
            Sentiment cached = cacheRegion != null ? cacheRegion.get(text) : null;
            if (cached != null) {
                results.add(cached);
            } else {
                results.add(null);
                misses.add(text);
            }
        }

        if (!misses.isEmpty()) {
            log.info("Executing onnx inference service using {} texts", misses.size());
            List<Sentiment> inferred = runInference(misses);
            int missIndex = 0;
            for (int i = 0; i < results.size(); i++) {
                if (results.get(i) == null) {
                    Sentiment value = inferred.get(missIndex++);
                    results.set(i, value);
                    if (cacheRegion != null) {
                        cacheRegion.put(texts.get(i), value);
                    }
                }
            }
        }

        return results;
    }

    protected List<Sentiment> runInference(List<String> texts) {
        ensureInitialized();
        Encoding[] encodings = tokenizer.batchEncode(texts.toArray(String[]::new));

        long[][] inputIds = Arrays.stream(encodings)
                .map(Encoding::getIds)
                .toArray(long[][]::new);

        long[][] attentionMask = Arrays.stream(encodings)
                .map(Encoding::getAttentionMask)
                .toArray(long[][]::new);

        try (OnnxTensor idsTensor = OnnxTensor.createTensor(env, inputIds);
             OnnxTensor maskTensor = OnnxTensor.createTensor(env, attentionMask);
             OrtSession.Result result = session.run(Map.of(
                     INPUT_IDS, idsTensor,
                     ATTENTION_MASK, maskTensor
             ))) {

            float[][] logitsBatch = (float[][]) result.get(0).getValue(); // [batch, 2]
            return Arrays.stream(logitsBatch)
                    .map(row -> toSentiment(row[0], row[1]))
                    .toList();
        } catch (OrtException e) {
            throw new UncheckedIOException(new IOException(e));
        }
    }

    private Sentiment toSentiment(float negLogit, float posLogit) {
        return posLogit >= negLogit
                ? Sentiment.POSITIVE
                : Sentiment.NEGATIVE;
    }

}
