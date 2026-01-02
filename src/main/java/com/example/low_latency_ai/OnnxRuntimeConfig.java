package com.example.low_latency_ai;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
class OnnxRuntimeConfig {

    @Value("${ai.model.path}")
    private String modelPath;

    @Bean
    OrtEnvironment ortEnvironment()
    {
        return OrtEnvironment.getEnvironment();
    }

    @Bean
    OrtSession ortSession(OrtEnvironment ortEnvironment) throws OrtException {
        return ortEnvironment.createSession(modelPath);
    }
}
