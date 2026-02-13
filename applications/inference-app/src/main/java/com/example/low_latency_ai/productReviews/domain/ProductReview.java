package com.example.low_latency_ai.productReviews.domain;

import lombok.Builder;
import org.springframework.data.gemfire.mapping.annotation.Region;

@Builder
@Region("ProductReviews")
public record ProductReview(String id,String productName, String review) {
}
