package com.example.low_latency_ai;

import ai.onnxruntime.OrtException;
import com.example.low_latency_ai.model.domain.AiModel;
import com.example.low_latency_ai.productReviews.domain.ProductReviewSummary;
import org.apache.geode.cache.DataPolicy;
import org.apache.geode.cache.client.ClientCache;
import org.springframework.context.annotation.Bean;
import org.springframework.data.gemfire.client.ClientRegionFactoryBean;
import org.springframework.data.gemfire.config.annotation.ClientCacheApplication;
import org.springframework.data.gemfire.config.annotation.EnableCachingDefinedRegions;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
@ClientCacheApplication(subscriptionEnabled = true)
@EnableCachingDefinedRegions
//@EnableGemfireFunctionExecutions
class GemFireConfig {

    @Bean
    ClientRegionFactoryBean<String, AiModel> aiModel(ClientCache gemfireClientCache) throws IOException, OrtException {
        var factory = new ClientRegionFactoryBean<String, AiModel>();
        factory.setCache(gemfireClientCache);
        factory.setName("AiModel");
        factory.setDataPolicy(DataPolicy.EMPTY);
        return factory;
    }


    @Bean
    ClientRegionFactoryBean<String, ProductReviewSummary> productReviews(ClientCache gemfireClientCache) throws IOException, OrtException {
        var factory = new ClientRegionFactoryBean<String, ProductReviewSummary>();
        factory.setCache(gemfireClientCache);
        factory.setName("ProductReviews");
        factory.setDataPolicy(DataPolicy.EMPTY);
        return factory;
    }
}
