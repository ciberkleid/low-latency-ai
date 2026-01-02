package com.example.low_latency_ai.service;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.example.low_latency_ai.domains.AiModel;
import com.example.low_latency_ai.domains.Sentiment;
import org.springframework.stereotype.Service;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@Service
class InferenceService {
    private static final String INPUT_IDS = "input_ids";
    private static final String ATTENTION_MASK = "attention_mask";

    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;

    public InferenceService(AiModel aiModel) throws OrtException, IOException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(aiModel.model(), new OrtSession.SessionOptions());
        this.tokenizer = HuggingFaceTokenizer.newInstance(new ByteArrayInputStream(aiModel.tokens()), Map.of());
    }

    /** Single-text, latency-optimized path (no batchEncode/streams). */
    public Sentiment execute(String text) throws OrtException {
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
            return toSentiment(logits[0], logits[1]);
        }
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

    private Sentiment toSentiment(float negLogit, float posLogit) {
        return posLogit >= negLogit
                ? Sentiment.POSITIVE
                : Sentiment.NEGATIVE;
    }
}
