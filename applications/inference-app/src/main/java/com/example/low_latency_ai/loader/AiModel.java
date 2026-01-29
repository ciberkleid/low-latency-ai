package com.example.low_latency_ai.loader;

import lombok.Builder;

import java.io.Serializable;

@Builder
public record AiModel(byte[] model, byte[] tokens) implements Serializable {
}
