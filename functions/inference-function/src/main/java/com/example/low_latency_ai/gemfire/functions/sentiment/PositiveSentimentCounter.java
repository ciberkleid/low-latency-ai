package com.example.low_latency_ai.gemfire.functions.sentiment;

import java.util.List;

@FunctionalInterface
public interface PositiveSentimentCounter {
    Long count(List<String> comments);
}
