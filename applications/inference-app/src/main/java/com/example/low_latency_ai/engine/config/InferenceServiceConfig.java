package com.example.low_latency_ai.engine.config;

import ai.onnxruntime.OrtException;
import com.example.low_latency_ai.engine.service.InferenceService;
import com.example.low_latency_ai.engine.service.OnnxInferenceService;
import org.apache.geode.cache.client.ClientCache;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.gemfire.GemfireTemplate;
import org.springframework.modulith.ApplicationModuleInitializer;

import java.io.IOException;

@Configuration
class InferenceServiceConfig {


    @Value("${ai.loader.key:sentiment}")
    private String modelKey;

    @Bean
    InferenceService onnxInferenceService(ClientCache clientCache, GemfireTemplate regionTemplate) throws IOException, OrtException {
        return new OnnxInferenceService(() -> regionTemplate.get(modelKey));
    }

    @Bean
    ApplicationModuleInitializer setupService(InferenceService service)
    {
        // Call service to ensure model is downloaded from GemFire when app is started
        return () -> service.execute("Making sure model is available in GemFire and pulled by engine module during startup.");
    }
}
