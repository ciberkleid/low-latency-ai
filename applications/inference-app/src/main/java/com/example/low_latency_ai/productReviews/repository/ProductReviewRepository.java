package com.example.low_latency_ai.productReviews.repository;

import com.example.low_latency_ai.domain.ProductReview;
import org.springframework.data.gemfire.repository.GemfireRepository;
import org.springframework.data.gemfire.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.Collection;

@Repository
public interface ProductReviewRepository extends GemfireRepository<ProductReview,String> {

    @Query("SELECT * FROM /ProductReviews p WHERE p.productName = $1")
    Collection<ProductReview> findByProductName(String productName);

}
