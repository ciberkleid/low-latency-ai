package com.example.low_latency_ai.productReviews.loader;

import com.example.low_latency_ai.productReviews.domain.ProductReview;
import com.example.low_latency_ai.productReviews.domain.ProductReviewSummary;
import com.example.low_latency_ai.productReviews.repository.ProductReviewRepository;
import lombok.RequiredArgsConstructor;
import nyla.solutions.core.io.csv.CsvReader;
import org.springframework.modulith.ApplicationModuleInitializer;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
class ProductReviewLoader implements ApplicationModuleInitializer {

    private final ProductReviewRepository repository;
    private final CsvReader csvReader;
    @Override
    public void initialize() {


        for(var lines : csvReader){
//            ProductReview productReview = new ProductReview(lines.get(0), lines.get(1), lines.get(2));
            ProductReview productReview = ProductReview.builder()
                    .id(lines.get(0))
                    .productName(lines.get(1))
                    .review(lines.get(2))
                    .build();

            repository.save(productReview);

        }
    }
}
