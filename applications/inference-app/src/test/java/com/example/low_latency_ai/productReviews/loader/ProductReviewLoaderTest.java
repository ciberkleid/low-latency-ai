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

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

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

    @Test
    void given_malformed_csv_row_when_load_then_throw_clear_error_and_do_not_save() throws IOException {
        var badCsv = """
                "1-spring", "spring"
                """;
        var badReader = new CsvReader(new StringReader(badCsv));
        var badSubject = new ProductReviewLoader(repository, badReader);

        assertThrows(IllegalArgumentException.class, badSubject::initialize);
        verifyNoInteractions(repository);
    }
}
