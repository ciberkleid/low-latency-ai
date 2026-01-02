package com.example.low_latency_ai.service;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.onnxruntime.*;
import com.example.low_latency_ai.domains.Sentiment;
import com.example.low_latency_ai.properties.SentimentProperties;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.util.Arrays;
import java.util.Map;

@Service
@RequiredArgsConstructor
class InferenceService {
    private final OrtSession session;
    private final OrtEnvironment env;
    private final HuggingFaceTokenizer tokenizer;
    private final SentimentProperties sentimentProperties;

    public Sentiment execute(String text) throws OrtException {

        String[] inputs = {text};

        Encoding[] encodings = tokenizer.batchEncode(inputs);

        long[][] inputIdsData = Arrays.stream(encodings)
                .map(Encoding::getIds)
                .toArray(long[][]::new);
        long[][] attentionMaskData = Arrays.stream(encodings)
                .map(Encoding::getAttentionMask)
                .toArray(long[][]::new);

        // Perform inference
        try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIdsData);
             OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMaskData)) {

            OrtSession.Result result = session.run(
                    Map.of("input_ids", inputIdsTensor,
                            "attention_mask", attentionMaskTensor),
                    session.getOutputNames()
            );

            float[][] outputData = (float[][]) result.get(0).getValue();



            float[] firstRow  =  outputData[0];

            float[] probabilities = softmax(firstRow);

            // Look at first index
            float sentimentFirstValue = probabilities[probabilities.length -1];
            if(sentimentFirstValue > sentimentProperties.postiveThreshold())
                return Sentiment.POSITIVE;
            else
                return Sentiment.NEGATIVE;
        }


    }

        private float[] softmax(float[] logits) {
            float maxLogit = logits[0];
            float sumExps = 0.0f;

            // Find the maximum logit for numerical stability
            for (float logit : logits) {
                if (logit > maxLogit) {
                    maxLogit = logit;
                }
            }

            // Compute exponentials and sum
            float[] exps = new float[logits.length];
            for (int i = 0; i < logits.length; i++) {
                exps[i] = (float) Math.exp(logits[i] - maxLogit);
                sumExps += exps[i];
            }

            // Normalize to get probabilities
            float[] softmax = new float[logits.length];
            for (int i = 0; i < logits.length; i++) {
                softmax[i] = exps[i] / sumExps;
            }

            return softmax;
        }

}
