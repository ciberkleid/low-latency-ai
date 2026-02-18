package com.example.low_latency_ai.engine.controller;

import com.example.low_latency_ai.domain.Sentiment;
import com.example.low_latency_ai.engine.service.InferenceService;
import lombok.SneakyThrows;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class InferenceRestControllerTest {
    private InferenceRestController subject;

    private String text = " I LOVE SPRING";
    @Mock
    private InferenceService service;

    @BeforeEach
    void setUp() {
        subject = new InferenceRestController(service);
    }

    @SneakyThrows
    @Test
    void checkSentiment() {

        Sentiment expected  = Sentiment.POSITIVE;

        when(service.execute(anyString())).thenReturn(expected);

        var actual = subject.checkSentiment(text);

        assertThat(actual).isEqualTo(expected);
    }
}