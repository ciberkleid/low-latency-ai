package com.example.low_latency_ai.model;

import lombok.Builder;

import java.io.Serializable;

@Builder
public record AiModel(byte[] model, byte[] tokens) implements Serializable {
}
