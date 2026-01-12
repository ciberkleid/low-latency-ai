package com.example.low_latency_ai.engine;

import ai.onnxruntime.OrtException;
import com.example.low_latency_ai.loader.AiModel;
import org.apache.geode.cache.DataPolicy;
import org.apache.geode.cache.client.ClientCache;
import org.springframework.context.annotation.Bean;
import org.springframework.data.gemfire.client.ClientRegionFactoryBean;
import org.springframework.data.gemfire.config.annotation.ClientCacheApplication;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
@ClientCacheApplication(subscriptionEnabled = true)
class GemFireConfig {

    @Bean
    ClientRegionFactoryBean<String, AiModel> AiModel(ClientCache gemfireClientCache) throws IOException, OrtException {
        var factory = new ClientRegionFactoryBean<String, AiModel>();
        factory.setCache(gemfireClientCache);
        factory.setName("AiModel");
        factory.setDataPolicy(DataPolicy.EMPTY);
        return factory;
    }
}
