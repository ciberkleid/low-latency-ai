package com.example.low_latency_ai.productReviews.loader;

import com.example.low_latency_ai.domain.ProductReview;
import com.example.low_latency_ai.productReviews.repository.ProductReviewRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import nyla.solutions.core.io.csv.CsvReader;
import org.springframework.modulith.ApplicationModuleInitializer;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
class ProductReviewLoader implements ApplicationModuleInitializer {

    private final ProductReviewRepository repository;
    private final CsvReader csvReader;
    @Override
    public void initialize() {

        int rowNumber = 0;
        for(var lines : csvReader){
            rowNumber++;
            if (lines == null || lines.size() < 3) {
                throw new IllegalArgumentException("Invalid product reviews CSV row at line " + rowNumber + ": expected at least 3 columns (id, productName, review)");
            }

            ProductReview productReview = ProductReview.builder()
                    .id(lines.get(0))
                    .productName(lines.get(1))
                    .review(lines.get(2))
                    .build();

            repository.save(productReview);

        }
        log.info("Loaded {} product reviews", rowNumber);
    }
}
