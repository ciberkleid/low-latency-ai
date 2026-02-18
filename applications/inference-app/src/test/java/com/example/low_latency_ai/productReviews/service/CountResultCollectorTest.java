package com.example.low_latency_ai.productReviews.service;

import org.apache.geode.distributed.DistributedMember;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;


@ExtendWith(MockitoExtension.class)
class CountResultCollectorTest {

    private CountResultCollector subject = new CountResultCollector();
    @Mock
    private DistributedMember member;


    @Test
    void given_func_results_from_a_member_when_add_result_then_sum_result() {

        Long expectedCount = 21L;
        subject.addResult(member,expectedCount);

        assertEquals(expectedCount,subject.countPositiveReviews());
    }
}