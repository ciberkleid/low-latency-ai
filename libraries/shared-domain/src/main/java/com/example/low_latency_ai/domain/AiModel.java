package com.example.low_latency_ai.domain;

import lombok.*;

@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class AiModel {
    byte[] model;
    byte[] tokens;
}
