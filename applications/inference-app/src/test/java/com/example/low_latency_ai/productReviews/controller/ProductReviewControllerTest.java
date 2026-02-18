package com.example.low_latency_ai.productReviews.controller;

import com.example.low_latency_ai.productReviews.domain.ProductReviewSummary;
import com.example.low_latency_ai.productReviews.service.ProductReviewCountFuncService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ProductReviewControllerTest {

    private final String productId = "spring";

    @Mock
    private ProductReviewCountFuncService service;
    private ProductReviewController subject;

    @BeforeEach
    void setUp() {
        subject = new ProductReviewController(service);
    }

    @Test
    void given_a_product_when_getProductReviews_then_return_percentage_positve_and_total() {
        var expected = ProductReviewSummary.builder().positiveCount(3).build(); ;
        when(service.countPositiveReviews(productId)).thenReturn(expected.positiveCount());

        var actual = subject.getProductReviewSummary(productId);
        assertThat(actual).isEqualTo(expected);
    }
}