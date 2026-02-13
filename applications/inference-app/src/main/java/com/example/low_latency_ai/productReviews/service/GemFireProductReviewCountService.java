package com.example.low_latency_ai.productReviews.service;

import com.example.low_latency_ai.productReviews.domain.ProductReviewSummary;
import lombok.RequiredArgsConstructor;
import org.apache.geode.cache.Region;
import org.springframework.stereotype.Service;

import java.util.Set;
import java.util.function.Supplier;

@Service
@RequiredArgsConstructor
public class GemFireProductReviewCountService implements ProductReviewCountFuncService{
    private final static String functionId = "countPositiveReviews";

    private final Region<String, ProductReviewSummary> productReviewsRegion;
    private final Supplier<CountResultCollector> resultCollectorProvider;
    private final ExecutionProvider executionProvider;


    @Override
    public long countPositiveReviews(String productId) {
        Set<?> productIs = Set.of(productId);

        CountResultCollector resultCollector = resultCollectorProvider.get();
        executionProvider.get(productReviewsRegion)
                .withFilter(productIs)
                .withCollector(resultCollector)
                .execute(functionId);
        return resultCollector.countPositiveReviews();
    }
}

