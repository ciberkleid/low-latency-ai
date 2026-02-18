package com.example.low_latency_ai.productReviews.config;

import com.example.low_latency_ai.productReviews.domain.ProductReviewSummary;
import com.example.low_latency_ai.productReviews.service.CountResultCollector;
import com.example.low_latency_ai.productReviews.service.ExecutionProvider;
import nyla.solutions.core.io.csv.CsvReader;
import org.apache.geode.cache.Region;
import org.apache.geode.cache.client.ClientCacheFactory;
import org.apache.geode.cache.execute.FunctionService;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.data.gemfire.GemfireTemplate;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

@Configuration
class ProductReviewsConfig {
    @Value("${product.reviews.path:classpath:/data/product-reviews.csv}")
    private String filePath;
    private final ResourceLoader resourceLoader;

    ProductReviewsConfig(ResourceLoader resourceLoader) {
        this.resourceLoader = resourceLoader;
    }

    @Bean
    Region<String, ProductReviewSummary> productReviewsRegion(@Qualifier("productReviewsTemplate") GemfireTemplate gemfireTemplate){

        return ClientCacheFactory.getAnyInstance().getRegion("ProductReviews");
    }

    @Bean
    Supplier<CountResultCollector> resultCollectorProvider(){
        return () -> new CountResultCollector();
    }

    @Bean
    ExecutionProvider executionProvider(){
        return region -> FunctionService.onRegion(region);
    }

    @Bean
    CsvReader csvReader() throws IOException {
        List<String> candidates = new ArrayList<>();
        candidates.add(filePath);

        Resource resource = null;
        String resolvedLocation = null;
        for (String candidate : candidates) {
            Resource candidateResource = resourceLoader.getResource(candidate);
            if (candidateResource.exists()) {
                resource = candidateResource;
                resolvedLocation = candidate;
                break;
            }
        }

        if (resource == null) {
            throw new IOException("CSV resource does not exist. Tried locations: " + candidates);
        }

        File csvFile;
        try {
            csvFile = resource.getFile();
        } catch (IOException notAFileResource) {
            // Classpath resources in packaged apps are not regular files; materialize for CsvReader.
            csvFile = File.createTempFile("product-reviews-", ".csv");
            csvFile.deleteOnExit();
            try (InputStream inputStream = resource.getInputStream()) {
                Files.copy(inputStream, csvFile.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
        }

        System.out.println("Using product reviews CSV from: " + resolvedLocation);
        return new CsvReader(csvFile);
    }

}
