package com.example.low_latency_ai.engine.service;

import com.example.low_latency_ai.model.domain.AiModel;
import com.example.low_latency_ai.model.domain.Sentiment;

public interface InferenceService {
    Sentiment execute(String text);

    void updateModel(AiModel aiModel);
}
