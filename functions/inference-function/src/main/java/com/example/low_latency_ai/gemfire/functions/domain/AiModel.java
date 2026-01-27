package com.example.low_latency_ai.gemfire.functions.domain;

import java.io.Serializable;

public record AiModel(byte[] model, byte[] tokens) implements Serializable {
}
