package com.example.low_latency_ai.model;

import lombok.SneakyThrows;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.Resource;
import org.springframework.data.gemfire.GemfireTemplate;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class AiModelLoaderTest {

    private AiModelLoader subject;
    @Mock
    private GemfireTemplate gemfireTemplate;
    private String key = "key";
    @Mock
    private Resource modelResource;
    @Mock
    private Resource tokenizer;

    @BeforeEach
    void setUp() {
        subject = new AiModelLoader(gemfireTemplate,key,modelResource,tokenizer);
    }

    @SneakyThrows
    @Test
    void given_gemfire_connection_when_run_then_load_AiModel() {

        subject.initialize();

        verify(gemfireTemplate).put(any(),any());

    }
}