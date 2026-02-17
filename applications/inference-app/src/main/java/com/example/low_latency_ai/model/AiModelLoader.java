package com.example.low_latency_ai.model;

import com.example.low_latency_ai.domain.AiModel;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.data.gemfire.GemfireTemplate;
import org.springframework.modulith.ApplicationModuleInitializer;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.UncheckedIOException;


@Component
class AiModelLoader implements ApplicationModuleInitializer {

    private final GemfireTemplate gemfireTemplate;
    private final String modelKey;

    private final Resource modelResource;
    private final Resource tokenizerResource;

    AiModelLoader(@Qualifier("aiModelTemplate") GemfireTemplate aiModelTemplate,
                  @Value("${ai.model.key:sentiment}")
                  String modelKey,
                  @Value("${ai.model.model:file:./models/model.onnx}")
                  Resource modelResource,
                  @Value("${ai.model.tokenizer:file:./models/tokenizer.json}")
                  Resource tokenizerResource) {
        this.gemfireTemplate = aiModelTemplate;
        this.modelKey = modelKey;
        this.modelResource = modelResource;
        this.tokenizerResource = tokenizerResource;
    }


    @Override
    public void initialize()  {

        try {
            gemfireTemplate.put(modelKey, AiModel.builder().model(modelResource.getContentAsByteArray())
                    .tokens(tokenizerResource.getContentAsByteArray()).build());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
