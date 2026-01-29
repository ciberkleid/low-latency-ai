package com.example.low_latency_ai.engine.bootstrap;

import com.example.low_latency_ai.loader.AiModel;
import org.apache.geode.cache.CacheListener;
import org.apache.geode.cache.Region;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.gemfire.GemfireTemplate;
import org.springframework.modulith.ApplicationModuleInitializer;
import org.springframework.stereotype.Component;

@Component
class AiModelUpdateListenerInitializer implements ApplicationModuleInitializer {

    private final ObjectProvider<CacheListener<String, AiModel>> listenerProvider;
    private final ObjectProvider<GemfireTemplate> regionProvider;

    AiModelUpdateListenerInitializer(ObjectProvider<CacheListener<String, AiModel>> listenerProvider,
                                     @Qualifier("aiModelTemplate")
                                     ObjectProvider<GemfireTemplate> regionProvider) {

        this.listenerProvider = listenerProvider;
        this.regionProvider = regionProvider;
    }

    @Override
    public void initialize() {

        Region<String,AiModel> region = regionProvider.getObject().getRegion();
        var listener = listenerProvider.getObject();

         region.getAttributesMutator().addCacheListener(listener);


    }
}
