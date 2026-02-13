package com.example.low_latency_ai.productReviews.domain;

import lombok.Builder;
import lombok.Data;

@Builder
public record ProductReviewSummary(String id, long positiveCount) {
}
