package com.example.low_latency_ai.productReviews.controller;

import com.example.low_latency_ai.productReviews.domain.ProductReviewSummary;
import com.example.low_latency_ai.productReviews.service.ProductReviewCountFuncService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("product/review")
class ProductReviewController {
    private final ProductReviewCountFuncService service;

    @GetMapping("{productId}")
    public ProductReviewSummary getProductReviewSummary(@PathVariable String productId) {
        return ProductReviewSummary.builder()
                .id(productId)
                .positiveCount(service.countPositiveReviews(productId))
                .build();
    }
}
