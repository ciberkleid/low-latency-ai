package com.example.low_latency_ai.engine.service;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.example.low_latency_ai.engine.domains.Sentiment;
import com.example.low_latency_ai.loader.AiModel;
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

    private void setupSessionAndTokenizer() {
        try {
            this.env = OrtEnvironment.getEnvironment();
            var aiModel = this.aiModelSupplier.get();

            this.session = env.createSession(aiModel.model(), new OrtSession.SessionOptions());
            this.tokenizer = HuggingFaceTokenizer.newInstance(new ByteArrayInputStream(aiModel.tokens()), Map.of());
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
        if(this.tokenizer == null) {
            this.setupSessionAndTokenizer();
        }
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
        this.aiModelSupplier = () -> aiModel;
        this.setupSessionAndTokenizer();
    }

    HuggingFaceTokenizer getTokenizer() {
        return this.tokenizer;
    }


//    /** Multi-text, throughput-optimized path (batchEncode + one ONNX call). */
//    public List<Sentiment> execute(List<String> texts) throws OrtException {
//        if (texts == null || texts.isEmpty()) {
//            return List.of();
//        }
//
//        Encoding[] encodings = tokenizer.batchEncode(texts.toArray(String[]::new));
//
//        long[][] inputIds = Arrays.stream(encodings)
//                .map(Encoding::getIds)
//                .toArray(long[][]::new);
//
//        long[][] attentionMask = Arrays.stream(encodings)
//                .map(Encoding::getAttentionMask)
//                .toArray(long[][]::new);
//
//        try (OnnxTensor idsTensor = OnnxTensor.createTensor(env, inputIds);
//             OnnxTensor maskTensor = OnnxTensor.createTensor(env, attentionMask);
//             OrtSession.Result result = session.run(Map.of(
//                     INPUT_IDS, idsTensor,
//                     ATTENTION_MASK, maskTensor
//             ))) {
//
//            float[][] logitsBatch = (float[][]) result.get(0).getValue(); // [batch, 2]
//            return Arrays.stream(logitsBatch)
//                    .map(row -> toSentiment(row[0], row[1]))
//                    .toList();
//        }
//    }
}
