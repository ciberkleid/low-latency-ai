package com.example.low_latency_ai.engine.service;

import com.example.low_latency_ai.loader.AiModel;
import com.example.low_latency_ai.engine.domains.Sentiment;

import java.io.IOException;

public interface InferenceService {
    Sentiment execute(String text) throws Exception;

    void updateModel(AiModel aiModel) throws IOException;
}
