package com.example.low_latency_ai.domains;

import java.io.Serializable;

public record AiModel(byte[] model, byte[] tokens) implements Serializable {
}
