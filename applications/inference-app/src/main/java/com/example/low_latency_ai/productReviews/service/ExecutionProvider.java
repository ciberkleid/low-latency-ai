package com.example.low_latency_ai.productReviews.service;

import com.example.low_latency_ai.domain.ProductReview;
import org.apache.geode.cache.Region;
import org.apache.geode.cache.execute.Execution;

@FunctionalInterface
public interface ExecutionProvider {
    Execution<Object, Object, Object> get(Region<String, ProductReview> productReviewsRegion);
}
