package com.example.low_latency_ai.engine;

import ai.onnxruntime.OrtException;
import com.example.low_latency_ai.engine.service.OnnxInferenceService;
import org.apache.geode.cache.client.ClientCache;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.data.gemfire.GemfireTemplate;

import java.io.IOException;

@Configuration
class ServiceConfig {


    @Value("${ai.loader.key:sentiment}")
    private String modelKey;

    @Bean
    OnnxInferenceService onnxInferenceService(ClientCache clientCache, GemfireTemplate regionTemplate) throws IOException, OrtException {
        return new OnnxInferenceService(() -> regionTemplate.get(modelKey));
    }

    @Bean
    @Order(2)
    CommandLineRunner setupService(OnnxInferenceService service)
    {
        return args -> service.execute("This is just a text for started to initialize the loader. This will failed is the loader is not loaded in GemFire");
    }
}
