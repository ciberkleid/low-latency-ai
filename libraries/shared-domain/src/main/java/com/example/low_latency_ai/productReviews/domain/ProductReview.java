package com.example.low_latency_ai.productReviews.domain;

import lombok.*;
import org.springframework.data.gemfire.mapping.annotation.Region;

@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Region("ProductReviews")
public class ProductReview {
    private String id;
    private String productName;
    private String review;
}