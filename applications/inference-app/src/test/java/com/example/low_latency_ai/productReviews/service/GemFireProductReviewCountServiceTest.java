package com.example.low_latency_ai.productReviews.service;

import com.example.low_latency_ai.productReviews.domain.ProductReviewSummary;
import org.apache.geode.cache.Region;
import org.apache.geode.cache.execute.Execution;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class GemFireProductReviewCountServiceTest {

    private GemFireProductReviewCountService subject;
    private final String productId = "spring";
    @Mock
    private Region<String, ProductReviewSummary> productReviewsRegion;
    @Mock
    private CountResultCollector collector;
    @Mock
    private ExecutionProvider provider;
    @Mock
    private Execution<Object, Object, Object> execution;

    @BeforeEach
    void setUp() {
        subject = new GemFireProductReviewCountService(productReviewsRegion,
                () -> collector,
                provider);
    }

    @Test
    void given_productId_when_countPositive_then_return_expected() {

        long expectedCount = 30;
        when(collector.countPositiveReviews()).thenReturn(expectedCount);
        when(provider.get(any())).thenReturn(execution);
        when(execution.withFilter(any())).thenReturn(execution);
        when(execution.withCollector(any())).thenReturn(execution);


        var actual = subject.countPositiveReviews(productId);

        Assertions.assertThat(actual).isEqualTo(expectedCount);

    }
}