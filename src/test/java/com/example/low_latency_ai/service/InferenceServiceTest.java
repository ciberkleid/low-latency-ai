package com.example.low_latency_ai.service;

import ai.onnxruntime.OrtException;
import com.example.low_latency_ai.domains.AiModel;
import com.example.low_latency_ai.domains.Sentiment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
class InferenceServiceTest {

    private static AiModel aiModel;

    private InferenceService subject;

    private String positiveText = "I love Spring";
    private String negativeText = "I do not like taxes";

    @BeforeAll
    static void beforeAll() throws OrtException, IOException {
        String onnxModelPath = "models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/model.onnx";
        String tokenPath = "models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tokenizer.json";

        var modelBytes = Files.readAllBytes(Paths.get(onnxModelPath));
        var tokenizerBytes = Files.readAllBytes(Paths.get(tokenPath));

        aiModel = new AiModel(modelBytes, tokenizerBytes);
    }

    @BeforeEach
    void setUp() throws IOException, OrtException {
        subject = new InferenceService(aiModel);
    }

    @Test
    void given_positive_text_when_execute_then_return_positive() throws OrtException {

        Sentiment actual = subject.execute(positiveText);
        assertThat(actual).isEqualTo(Sentiment.POSITIVE);
    }

    @Test
    void given_negative_text_when_execute_then_return_negative() throws OrtException {

        Sentiment actual = subject.execute(negativeText);
        assertThat(actual).isEqualTo(Sentiment.NEGATIVE);
    }

}