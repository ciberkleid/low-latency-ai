package com.example.low_latency_ai.productReviews.service;

import lombok.extern.slf4j.Slf4j;
import org.apache.geode.cache.execute.ResultCollector;
import org.apache.geode.distributed.DistributedMember;
import org.springframework.geode.function.support.AbstractResultCollector;

import java.util.List;

@Slf4j
public class CountResultCollector extends AbstractResultCollector<Object,
java.lang.Object> {

    private Long count = 0L;

    public long countPositiveReviews() {
        return count;
    }

    @Override
    public void addResult(DistributedMember distributedMember, Object positiveReviewsCountFromAFuncInMember) {

        log.info("Adding results for distributed member: {}", distributedMember);
        count += (Long)positiveReviewsCountFromAFuncInMember;
    }
}
