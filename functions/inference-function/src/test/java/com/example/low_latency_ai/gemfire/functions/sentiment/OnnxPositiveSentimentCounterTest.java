package com.example.low_latency_ai.gemfire.functions.sentiment;

import com.example.low_latency_ai.gemfire.functions.domain.AiModel;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Supplier;

class OnnxPositiveSentimentCounterTest {

    private static AiModel aiModel;

    private OnnxPositiveSentimentCounter subject;

    private Supplier<AiModel> aiModelSupplier;


    @BeforeAll
    static void beforeAll() throws IOException {
        String onnxModelPath = "../../models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/model.onnx";
        String tokenPath = "../../models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tokenizer.json";

        var modelBytes = Files.readAllBytes(Paths.get(onnxModelPath));
        var tokenizerBytes = Files.readAllBytes(Paths.get(tokenPath));

        aiModel = new AiModel(modelBytes, tokenizerBytes);
    }

    @BeforeEach
    void setUp() {
        aiModelSupplier = () -> aiModel;
        subject = new OnnxPositiveSentimentCounter(aiModelSupplier);
    }

    @Test
    void given_list_comments_with_count_use_aiModel_count_positives() {
        var mixedComments = List.of(
                "I love Spring",
                "Speak for yourself, I do NOT like it",
                "Absolutely fantastic"
        );

        var actual = subject.count(mixedComments);

        Assertions.assertEquals(2L, actual);
    }

    @Test
    void given_positive_only_comments_when_count_then_return_all() {
        var positiveOnly = List.of(
                "I love Spring",
                "This is excellent",
                "Absolutely fantastic"
        );

        var actual = subject.count(positiveOnly);

        Assertions.assertEquals(3L, actual);
    }

    @Test
    void given_negative_only_comments_when_count_then_return_zero() {
        var negativeOnly = List.of(
                "I do not like it",
                "This is terrible",
                "Absolutely awful"
        );

        var actual = subject.count(negativeOnly);

        Assertions.assertEquals(0L, actual);
    }
}
