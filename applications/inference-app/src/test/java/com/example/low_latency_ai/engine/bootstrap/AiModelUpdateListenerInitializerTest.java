package com.example.low_latency_ai.engine.bootstrap;

import com.example.low_latency_ai.domain.AiModel;
import org.apache.geode.cache.AttributesMutator;
import org.apache.geode.cache.CacheListener;
import org.apache.geode.cache.Region;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.data.gemfire.GemfireTemplate;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AiModelUpdateListenerInitializerTest {


    private AiModelUpdateListenerInitializer subject;
    @Mock
    private Region<String, AiModel> aiModelRegion;

    @Mock
    private ObjectProvider<CacheListener<String,AiModel>> listenerProvider;
    @Mock
    private ObjectProvider<GemfireTemplate> regionProvider;
    @Mock
    private AttributesMutator<String, AiModel> attributesMutator;
    @Mock
    private CacheListener<String, AiModel> listener;
    @Mock
    private GemfireTemplate gemfireTemplate;

    @BeforeEach
    void setUp() {
        subject = new AiModelUpdateListenerInitializer(listenerProvider, regionProvider);
    }

    @Test
    void given_aiModel_region_when_initialize_register_listener_on_region() {

        when(regionProvider.getObject()).thenReturn(gemfireTemplate);
        when(gemfireTemplate.getRegion()).thenReturn((Region)aiModelRegion);

        when(listenerProvider.getObject()).thenReturn(listener);
        when(aiModelRegion.getAttributesMutator()).thenReturn(attributesMutator);

        subject.initialize();

        //verify the region has a listener registered
        verify(attributesMutator).addCacheListener(any());
    }
}