package com.example.low_latency_ai.productReviews.domain;

import lombok.Builder;

@Builder
public record ProductReviewSummary(String id, long positiveCount) {
}
