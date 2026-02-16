package com.example.low_latency_ai.engine.service;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.example.low_latency_ai.model.domain.AiModel;
import com.example.low_latency_ai.model.domain.Sentiment;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.geode.cache.EntryEvent;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.geode.cache.AbstractCommonEventProcessingCacheListener;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Map;
import java.util.function.Supplier;

@Slf4j
public class OnnxInferenceService extends AbstractCommonEventProcessingCacheListener<String, AiModel> implements InferenceService {
    private static final String INPUT_IDS = "input_ids";
    private static final String ATTENTION_MASK = "attention_mask";
    private Supplier<AiModel> aiModelSupplier;

    private OrtEnvironment env;
    private OrtSession session;
    private HuggingFaceTokenizer tokenizer;

    public OnnxInferenceService(Supplier<AiModel >aiModelSupplier){
        this.aiModelSupplier = aiModelSupplier;
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

    /** Single-text, latency-optimized path (no batchEncode/streams). */
    @Override
    @Cacheable("SentimentResults")
    public Sentiment execute(String text) {

        log.info("Executing onnx inference service using text: {}",text);
        ensureInitialized();
        Encoding enc = tokenizer.encode(text);

        long[][] inputIds = { enc.getIds() };
        long[][] attentionMask = { enc.getAttentionMask() };

        try (OnnxTensor idsTensor = OnnxTensor.createTensor(env, inputIds);
             OnnxTensor maskTensor = OnnxTensor.createTensor(env, attentionMask);
             OrtSession.Result result = session.run(Map.of(
                     INPUT_IDS, idsTensor,
                     ATTENTION_MASK, maskTensor
             ))) {

            float[] logits = ((float[][]) result.get(0).getValue())[0]; // [neg, pos]
            var results =  toSentiment(logits[0], logits[1]);

            log.info("Sentiments results: {}", results);
            return results;
        } catch (OrtException e) {
            throw new UncheckedIOException(new IOException(e));
        }
    }

    @SneakyThrows
    @Override
    protected void processEntryEvent(EntryEvent<String, AiModel> event, AbstractCommonEventProcessingCacheListener.EntryEventType eventType) {
        log.info("Update event detected: Received new model from GemFire.");
        if(AbstractCommonEventProcessingCacheListener.EntryEventType.DESTROY.equals(eventType) || AbstractCommonEventProcessingCacheListener.EntryEventType.INVALIDATE.equals(eventType))
            return;

        updateModel(event.getNewValue());
    }

    private Sentiment toSentiment(float negLogit, float posLogit) {
        return posLogit >= negLogit
                ? Sentiment.POSITIVE
                : Sentiment.NEGATIVE;
    }

    @Override
    public void updateModel(AiModel aiModel) {
        log.info("Updating local Onnx session and tokenizer with new model.");
        synchronized (this) {
            this.aiModelSupplier = () -> aiModel; // Update Supplier with new model
            this.setupSessionAndTokenizer(); // Recreate session and tokenizer using updated model
        }
    }

    HuggingFaceTokenizer getTokenizer() {
        return this.tokenizer;
    }

}
