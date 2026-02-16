package com.example.low_latency_ai.gemfire.functions.sentiment;

import com.example.low_latency_ai.model.domain.AiModel;
import com.example.low_latency_ai.model.domain.Sentiment;
import org.apache.geode.cache.Region;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OnnxPositiveSentimentCounterCacheTest {

    @Test
    void given_cached_and_uncached_texts_when_execute_then_hit_cache_and_store_miss() {
        Region<String, Sentiment> region = mock(Region.class);
        when(region.get("good")).thenReturn(Sentiment.POSITIVE);
        when(region.get("bad")).thenReturn(null);

        var subject = new TestCounter(region);

        var results = subject.execute(List.of("good", "bad"));

        Assertions.assertEquals(List.of(Sentiment.POSITIVE, Sentiment.NEGATIVE), results);
        Assertions.assertEquals(1, subject.inferenceCalls.get());
        Assertions.assertEquals(List.of("bad"), subject.inferenceTexts);
        verify(region).put("bad", Sentiment.NEGATIVE);
    }

    private static final class TestCounter extends OnnxPositiveSentimentCounter {
        private final Region<String, Sentiment> region;
        private final AtomicInteger inferenceCalls = new AtomicInteger();
        private List<String> inferenceTexts;

        private TestCounter(Region<String, Sentiment> region) {
            super((Supplier<AiModel>) () -> null);
            this.region = region;
        }

        @Override
        protected Region<String, Sentiment> getSentimentRegion() {
            return region;
        }

        @Override
        protected List<Sentiment> runInference(List<String> texts) {
            inferenceCalls.incrementAndGet();
            inferenceTexts = texts;
            return texts.stream()
                    .map(text -> text.contains("bad") ? Sentiment.NEGATIVE : Sentiment.POSITIVE)
                    .toList();
        }
    }
}
