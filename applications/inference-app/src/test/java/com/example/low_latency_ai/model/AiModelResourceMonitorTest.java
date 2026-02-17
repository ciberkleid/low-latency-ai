package com.example.low_latency_ai.model;

import com.example.low_latency_ai.domain.AiModel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.Resource;
import org.springframework.data.gemfire.GemfireTemplate;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AiModelResourceMonitorTest {

    @Mock
    private GemfireTemplate gemfireTemplate;
    @Mock
    private Resource modelResource;
    @Mock
    private Resource tokenizerResource;

    private AiModelResourceMonitor subject;

    @BeforeEach
    void setUp() {
        subject = new AiModelResourceMonitor(gemfireTemplate, "key", modelResource, tokenizerResource);
    }

    @Test
    void when_resources_change_then_refreshes_gemfire_entry() throws Exception {
        // simulate model file change: first poll sees 1L, second poll sees 2L
        when(modelResource.lastModified()).thenReturn(1L, 2L);
        when(tokenizerResource.lastModified()).thenReturn(1L, 1L);
        when(modelResource.getContentAsByteArray()).thenReturn(new byte[]{0x1});
        when(tokenizerResource.getContentAsByteArray()).thenReturn(new byte[]{0x2});

        subject.captureInitialTimestamps();

        subject.checkForUpdates();

        ArgumentCaptor<AiModel> modelCaptor = ArgumentCaptor.forClass(AiModel.class);
        verify(gemfireTemplate).put(eq("key"), modelCaptor.capture());
        assertArrayEquals(new byte[]{0x1}, modelCaptor.getValue().getModel());
        assertArrayEquals(new byte[]{0x2}, modelCaptor.getValue().getTokens());
    }

    @Test
    void when_no_change_detected_then_does_not_write_to_gemfire() throws Exception {
        // Both polls see the same timestamps => no model or tokenizer file changes detected
        when(modelResource.lastModified()).thenReturn(5L, 5L);
        when(tokenizerResource.lastModified()).thenReturn(5L, 5L);

        subject.captureInitialTimestamps();

        subject.checkForUpdates();

        verify(gemfireTemplate, never()).put(any(), any());
    }
}
