package com.example.low_latency_ai.gemfire.functions;

import com.example.low_latency_ai.domain.AiModel;
import com.example.low_latency_ai.domain.ProductReview;
import com.example.low_latency_ai.gemfire.functions.sentiment.OnnxPositiveSentimentCounter;
import com.example.low_latency_ai.gemfire.functions.sentiment.PositiveSentimentCounter;
import org.apache.geode.cache.CacheFactory;
import org.apache.geode.cache.Region;
import org.apache.geode.cache.execute.Function;
import org.apache.geode.cache.execute.FunctionContext;
import org.apache.geode.cache.execute.FunctionException;
import org.apache.geode.cache.execute.RegionFunctionContext;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collection;

public class CountPositiveProductReviewsFunction implements Function<String[]> {

    private static final String REGION_NM = "AiModel";
    private static final String MODEL_KEY_NM = "sentiment";
    private String selectByProductReviewQuery = """
            select review from /ProductReviews where productName = $1
            """;
    private final PositiveSentimentCounter positiveSentimentCounter;
    private Logger logger = LogManager.getLogger(CountPositiveProductReviewsFunction.class);

    // No args constructor is needed by each GemFire server to initialize
    // the function at time of deployment
    public CountPositiveProductReviewsFunction(){
        this(new OnnxPositiveSentimentCounter(() ->{
            Region<String, AiModel> region = CacheFactory.getAnyInstance()
                    .getRegion(REGION_NM);

            return region.get(MODEL_KEY_NM);
        }));
    }

    public CountPositiveProductReviewsFunction(PositiveSentimentCounter positiveSentimentCounter) {
        this.positiveSentimentCounter = positiveSentimentCounter;
    }

    @Override
    public void execute(FunctionContext<String[]> functionContext) {

        logger.info("Executing function");
        // Notes:
        // 1. Use FunctionContext<String[]> rather than FunctionContext<String> to
        //    facilitate testing from gfsh, as gfsh passes a String[] of arguments
        // 2. The Region to act on is part of the passed functionContext

        var rfc = (RegionFunctionContext) functionContext;

        Region<String, ProductReview> productReviewsRegion = rfc.getDataSet();

        // String[] productName = {(String) rfc.getFilter().iterator().next()};
        String productName = (String) rfc.getArguments(); // The argument here will be the productName

        logger.info("Product Name: {}",productName);
        try {
            var query = rfc.getCache().getQueryService().newQuery(selectByProductReviewQuery);

            logger.info("Query: {}",query);

            // TODO: This may require using a org.apache.geode.cache.query.Struct instead of a Collection of Strings
            Collection<String> productReviews = (Collection)query.execute(rfc, new Object[]{productName});

            logger.info("Results: {}",productReviews.size());
            productReviews.stream().forEach(productReview -> logger.info("Product: {}",productReview));
            var count = positiveSentimentCounter.count(new ArrayList<String>(productReviews));
            logger.info("count: {}",count);

            rfc.getResultSender().lastResult(count);

        }
        catch (Exception e) {
            throw new FunctionException(e);
        }

    }

    // This determines the name of the function on GemFire
    @Override
    public String getId() {
        return "countPositiveReviews";
    }
}
