package com.example.low_latency_ai.loader;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.core.io.Resource;
import org.springframework.data.gemfire.GemfireTemplate;
import org.springframework.stereotype.Component;


@Component
@Order(1)
class AiModelLoader implements CommandLineRunner {

    private final GemfireTemplate gemfireTemplate;
    private final String modelKey;

    private final Resource modelResource;
    private final Resource tokenizerResource;

    AiModelLoader(GemfireTemplate gemfireTemplate,
                  @Value("${ai.loader.key:sentiment}")
                  String modelKey,
//                  @Value("${/Users/ciberkleid/workspace/devnexus/low-latency-ai/src/main/resources/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/model.onnx}")
                  @Value("classpath:/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/model.onnx")
                  Resource modelResource,
                  @Value("classpath:/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tokenizer.json")
                  Resource tokenizerResource) {
        this.gemfireTemplate = gemfireTemplate;
        this.modelKey = modelKey;
        this.modelResource = modelResource;
        this.tokenizerResource = tokenizerResource;
    }


    @Override
    public void run(String... args) throws Exception {
        gemfireTemplate.put(modelKey, AiModel.builder().model(modelResource.getContentAsByteArray())
                .tokens(tokenizerResource.getContentAsByteArray()).build());
    }
}
