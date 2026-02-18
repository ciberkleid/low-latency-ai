package com.example.low_latency_ai.engine.service;

import ai.onnxruntime.OrtException;
import com.example.low_latency_ai.domain.AiModel;
import com.example.low_latency_ai.domain.Sentiment;
import org.apache.geode.cache.EntryEvent;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.geode.cache.AbstractCommonEventProcessingCacheListener;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class OnnxInferenceServiceTest {

    private static AiModel aiModel;

    private OnnxInferenceService subject;

    private String positiveText = "I love Spring";
    private String negativeText = "I do not like taxes";
    @Mock
    private EntryEvent<String, AiModel> event;


    @BeforeAll
    static void beforeAll() throws OrtException, IOException {
        String onnxModelPath = "../../models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/model.onnx";
        String tokenPath = "../../models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tokenizer.json";

        var modelBytes = Files.readAllBytes(Paths.get(onnxModelPath));
        var tokenizerBytes = Files.readAllBytes(Paths.get(tokenPath));

        aiModel = new AiModel(modelBytes, tokenizerBytes);
    }

    @BeforeEach
    void setUp() throws IOException, OrtException {
        subject = new OnnxInferenceService(() -> aiModel);
    }

    @Test
    void given_positive_text_when_execute_then_return_positive() throws OrtException, IOException {

        Sentiment actual = subject.execute(positiveText);
        assertThat(actual).isEqualTo(Sentiment.POSITIVE);
    }

    @Test
    void given_negative_text_when_execute_then_return_negative() throws OrtException, IOException {

        Sentiment actual = subject.execute(negativeText);
        assertThat(actual).isEqualTo(Sentiment.NEGATIVE);
    }

    @Test
    void given_aiModel_when_update_Model_then_update_session() {

        assertDoesNotThrow(() -> subject.updateModel(aiModel));
    }

    @Test
    void given_model_when_updateModel_then_updateTokenizer() throws IOException {

        var tokenizer = subject.getTokenizer();
        subject.updateModel(aiModel);

        assertThat(tokenizer).isNotEqualTo(subject.getTokenizer());


    }

    @Test
    void given_create_model_when_update_update_the_service() throws IOException {

        var eventType = AbstractCommonEventProcessingCacheListener.EntryEventType.CREATE;

        when(event.getNewValue()).thenReturn(aiModel);

        subject.processEntryEvent(event, eventType);

        // Verify that updateModel was called
//        verify(service).updateModel(any(AiModel.class));
    }

    @Test
    void given_update_model_when_update_update_the_service() throws IOException {

        var eventType = AbstractCommonEventProcessingCacheListener.EntryEventType.UPDATE;

        when(event.getNewValue()).thenReturn(aiModel);

        subject.processEntryEvent(event, eventType);

        // Verify that updateModel was called
//        verify(service).updateModel(any(AiModel.class));
    }

    @Test
    void given_a_destroy_event_when_update_then_do_not_update() throws IOException {
        var eventType = AbstractCommonEventProcessingCacheListener.EntryEventType.DESTROY;

        subject.processEntryEvent(event, eventType);

        // Verify that updateModel was not called
//        verify(service,times(0)).updateModel(any(AiModel.class));
    }

    @Test
    void given_a_invalidate_event_when_update_then_do_not_update() throws IOException {
        var eventType = AbstractCommonEventProcessingCacheListener.EntryEventType.INVALIDATE;

        subject.processEntryEvent(event, eventType);

        // Verify that updateModel was not called
//        verify(service,times(0)).updateModel(any(AiModel.class));
    }
}