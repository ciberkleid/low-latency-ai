package com.example.low_latency_ai.model;

import com.example.low_latency_ai.model.domain.AiModel;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.data.gemfire.GemfireTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Watches the model and tokenizer resources for changes and refreshes the
 * GemFire cache entry when either file changes.
 */
@Slf4j
@Component
class AiModelResourceMonitor {

    private final GemfireTemplate gemfireTemplate;
    private final String modelKey;
    private final Resource modelResource;
    private final Resource tokenizerResource;

    private volatile Long lastModelModified;
    private volatile Long lastTokenizerModified;

    AiModelResourceMonitor(@Qualifier("aiModelTemplate") GemfireTemplate aiModelTemplate,
                           @Value("${ai.model.key:sentiment}") String modelKey,
                           @Value("${ai.model.model:classpath:/models/model.onnx}") Resource modelResource,
                           @Value("${ai.model.tokenizer:classpath:/models/tokenizer.json}") Resource tokenizerResource) {
        this.gemfireTemplate = aiModelTemplate;
        this.modelKey = modelKey;
        this.modelResource = modelResource;
        this.tokenizerResource = tokenizerResource;
    }

    @PostConstruct
    void captureInitialTimestamps() {
        lastModelModified = safeLastModified(modelResource);
        lastTokenizerModified = safeLastModified(tokenizerResource);
    }

    /**
     * Polls for changes on the configured resources.
     * Interval is configurable via ai.model.watch-interval (milliseconds).
     */
    @Scheduled(fixedDelayString = "${ai.model.watch-interval:5000}")
    void checkForUpdates() {
        Long currentModelModified = safeLastModified(modelResource);
        Long currentTokenizerModified = safeLastModified(tokenizerResource);

        boolean modelChanged = hasChanged(lastModelModified, currentModelModified);
        boolean tokenizerChanged = hasChanged(lastTokenizerModified, currentTokenizerModified);

        if (modelChanged || tokenizerChanged) {
            log.info("Detected change in AI assets (model: {}, tokenizer: {}). Refreshing GemFire entry.",
                    modelChanged, tokenizerChanged);
            refreshModelInGemFire();
            lastModelModified = currentModelModified;
            lastTokenizerModified = currentTokenizerModified;
        }
    }

    private boolean hasChanged(Long previous, Long current) {
        if (previous == null && current != null) {
            return true;
        }
        if (previous != null) {
            return !previous.equals(current);
        }
        return false;
    }

    // If lastModified cannot be read (e.g., resource packaged in a JAR), monitoring
    // for that resource is skipped and a warning is logged.
    private Long safeLastModified(Resource resource) {
        try {
            return resource.lastModified();
        } catch (IOException e) {
            log.warn("Unable to read lastModified for resource {}. Change detection disabled for this resource.", resource, e);
            return null;
        }
    }

    private void refreshModelInGemFire() {
        try {
            gemfireTemplate.put(modelKey, AiModel.builder()
                    .model(modelResource.getContentAsByteArray())
                    .tokens(tokenizerResource.getContentAsByteArray())
                    .build());
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to refresh AI model in GemFire", e);
        }
    }
}
