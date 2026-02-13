package com.example.low_latency_ai.gemfire.functions;

import com.example.low_latency_ai.gemfire.functions.domain.ProductReview;
import com.example.low_latency_ai.gemfire.functions.sentiment.PositiveSentimentCounter;
import org.apache.geode.cache.Region;
import org.apache.geode.cache.RegionService;
import org.apache.geode.cache.execute.RegionFunctionContext;
import org.apache.geode.cache.execute.ResultSender;
import org.apache.geode.cache.query.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class CountPositiveProductReviewsFunctionTest {

    @Mock
    private PositiveSentimentCounter positiveSentimentCounter;

    @Mock
    private RegionFunctionContext<String[]> rfc;
    @Mock
    private Region<String, ProductReview> productReviewsRegion;
    @Mock
    private ResultSender resultSender;
    @Mock
    private RegionService regionService;
    @Mock
    private QueryService queryService;
    @Mock
    private Query query;

    private CountPositiveProductReviewsFunction subject;
    private String productId = "Spring";

    @BeforeEach
    void setUp() {
        subject = new CountPositiveProductReviewsFunction(positiveSentimentCounter);
    }

    @Test
    void given_product_when_execute_then_return_positive_sentiment_count() throws FunctionDomainException, TypeMismatchException, QueryInvocationTargetException, NameResolutionException {

        var listOfProductComments = List.of("I love Junit", "This is a lot of mocking");
        Long positiveCount = 1L;

        Set productIdArg = Set.of(productId);
        when(rfc.getFilter()).thenReturn(productIdArg);

        when(rfc.getResultSender()).thenReturn(resultSender);

        when(rfc.getDataSet()).thenReturn((Region)productReviewsRegion);
        when(productReviewsRegion.getRegionService()).thenReturn(regionService);
        when(regionService.getQueryService()).thenReturn(queryService);
        when(queryService.newQuery(any())).thenReturn(query);
        when(query.execute(any(RegionFunctionContext.class),any())).thenReturn(listOfProductComments);

        when(positiveSentimentCounter.count(any())).thenReturn(positiveCount);

        subject.execute(rfc);

        Mockito.verify(resultSender).lastResult(any(Long.class));
    }
}