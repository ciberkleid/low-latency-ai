package com.example.low_latency_ai.engine.service;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
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
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
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
    void given_existing_runtime_objects_when_updateModel_then_previous_session_and_tokenizer_are_closed() throws Exception {
        // Initialize model runtime resources (session + tokenizer) via a normal inference request.
        subject.execute(positiveText);
        OrtSession oldSession = subject.getSession();
        var oldTokenizer = subject.getTokenizer();

        // Trigger model refresh; this should swap in new resources and close the old ones.
        subject.updateModel(aiModel);

        assertThat(subject.getSession()).isNotSameAs(oldSession);
        assertThat(subject.getTokenizer()).isNotSameAs(oldTokenizer);

        // Closed tokenizer should not accept further encode operations.
        assertThatThrownBy(() -> oldTokenizer.encode("still-usable?"))
                .as("Old tokenizer should be closed after updateModel")
                .isInstanceOfAny(Exception.class);

        // Closed session should not allow inference calls.
        Encoding encoding = subject.getTokenizer().encode("still-usable?");
        long[][] inputIds = {encoding.getIds()};
        long[][] attentionMask = {encoding.getAttentionMask()};
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        try (OnnxTensor idsTensor = OnnxTensor.createTensor(env, inputIds);
             OnnxTensor maskTensor = OnnxTensor.createTensor(env, attentionMask)) {
            assertThatThrownBy(() -> oldSession.run(Map.of(
                    "input_ids", idsTensor,
                    "attention_mask", maskTensor
            )))
                    .as("Old ONNX session should be closed after updateModel")
                    .isInstanceOfAny(Exception.class);
        }
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
