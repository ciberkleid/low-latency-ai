package com.example.low_latency_ai.productReviews.domain;

import lombok.Builder;
import lombok.Data;
import org.springframework.modulith.NamedInterface;

//@NamedInterface
@Builder
public record ProductReviewSummary(String id, long positiveCount) {
}
