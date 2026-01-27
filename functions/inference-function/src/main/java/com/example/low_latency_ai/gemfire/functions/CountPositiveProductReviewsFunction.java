package com.example.low_latency_ai.gemfire.functions;

import com.example.low_latency_ai.gemfire.functions.domain.ProductReview;
import com.example.low_latency_ai.gemfire.functions.sentiment.PositiveSentimentCounter;
import org.apache.geode.cache.Region;
import org.apache.geode.cache.execute.Function;
import org.apache.geode.cache.execute.FunctionContext;
import org.apache.geode.cache.execute.FunctionException;
import org.apache.geode.cache.execute.RegionFunctionContext;

import java.util.ArrayList;
import java.util.Collection;

public class CountPositiveProductReviewsFunction implements Function<String[]> {


    private String selectByProductReviewQuery = """
            select comment from /ProductReviews where productId = $1
            """;
    private final PositiveSentimentCounter positiveSentimentCounter;

    public CountPositiveProductReviewsFunction(PositiveSentimentCounter positiveSentimentCounter) {
        this.positiveSentimentCounter = positiveSentimentCounter;
    }

    @Override
    public void execute(FunctionContext<String[]> functionContext) {
        // Notes:
        // 1. Use FunctionContext<String[]> rather than FunctionContext<String> to
        //    facilitate testing from gfsh, as gfsh passes a String[] of arguments
        // 2. The Region to act on is part of the passed functionContext

        var rfc = (RegionFunctionContext) functionContext;

        Region<String, ProductReview> productReviewsRegion = rfc.getDataSet();


        var productId = functionContext.getArguments()[0];
        try {
            var query = productReviewsRegion.getRegionService().getQueryService().newQuery(selectByProductReviewQuery);

            // TODO: This may require using a org.apache.geode.cache.query.Struct
            // Instead of a Collection of Strings
            Collection<String> productReviews = (Collection)query.execute(rfc,productId);
            var count = positiveSentimentCounter.count(new ArrayList<String>(productReviews));

            rfc.getResultSender().lastResult(count);

        }
        catch (Exception e) {
            throw new FunctionException(e);
        }

    }
}
