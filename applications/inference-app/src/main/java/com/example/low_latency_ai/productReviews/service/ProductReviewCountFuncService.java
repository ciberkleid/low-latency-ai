package com.example.low_latency_ai.productReviews.service;

import org.springframework.data.gemfire.function.annotation.Filter;
import org.springframework.data.gemfire.function.annotation.OnRegion;

@OnRegion( region = "ProductReviews")
public interface ProductReviewCountFuncService {
    long countPositiveReviews(@Filter String productId);
}
