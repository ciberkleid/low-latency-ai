package com.example.low_latency_ai.engine.service;

import com.example.low_latency_ai.domain.AiModel;
import com.example.low_latency_ai.domain.Sentiment;

public interface InferenceService {
    Sentiment execute(String text);

    void updateModel(AiModel aiModel);
}
