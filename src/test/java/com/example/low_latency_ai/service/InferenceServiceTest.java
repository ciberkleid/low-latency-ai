package com.example.low_latency_ai.service;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.example.low_latency_ai.domains.Sentiment;
import com.example.low_latency_ai.properties.SentimentProperties;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.IOException;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
class InferenceServiceTest {

    private static OrtEnvironment env;
    private static OrtSession session;
    private static HuggingFaceTokenizer tokenizer;


    private InferenceService subject;
    private String positiveText = "I love Spring";
    private SentimentProperties properties;
    private float positiveThreshold = 0.6f;
    private float negativeThreshold = 0.4f;
    private String negativeText = "I do not like Taxes";
    private String neutralText = "Today is Today";

    @BeforeAll
    static void beforeAll() throws OrtException, IOException {
        String onnxModelPath = "models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/model.onnx";
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        session = env.createSession(onnxModelPath, sessionOptions);
        tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tokenizer.json"));
    }

    @BeforeEach
    void setUp() {
        properties = new SentimentProperties(positiveThreshold);
        subject = new InferenceService(session,env,tokenizer,properties);
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