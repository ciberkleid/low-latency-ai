package com.example.low_latency_ai.loader;

import lombok.Builder;
import org.springframework.modulith.NamedInterface;

import java.io.Serializable;

//@NamedInterface
@Builder
public record AiModel(byte[] model, byte[] tokens) implements Serializable {
}
