package com.example.low_latency_ai.productReviews.repository;

import com.example.low_latency_ai.productReviews.domain.ProductReview;
import org.springframework.data.gemfire.repository.GemfireRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ProductReviewRepository extends GemfireRepository<ProductReview,String> {
}
