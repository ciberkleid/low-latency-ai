package com.example.low_latency_ai.productReviews.loader;

import com.example.low_latency_ai.domain.ProductReview;
import com.example.low_latency_ai.productReviews.repository.ProductReviewRepository;
import nyla.solutions.core.io.csv.CsvReader;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.IOException;
import java.io.StringReader;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class ProductReviewLoaderTest {


    private ProductReviewLoader subject;
    @Mock
    private ProductReviewRepository repository;

    private CsvReader csvReader;

    private String csvData = """
            "1-spring", "spring", "I love Spring"
            """;


    @BeforeEach
    void setUp() throws IOException {
        csvReader = new CsvReader(new StringReader(csvData));
        subject = new ProductReviewLoader(repository,csvReader);
    }

    @Test
    void load_product_reviews() {

        subject.initialize();

        verify(repository).save(any(ProductReview.class));

    }
}