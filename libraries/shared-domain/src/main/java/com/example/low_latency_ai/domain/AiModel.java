package com.example.low_latency_ai.domain;

public class AiModel {
    private byte[] model;
    private byte[] tokens;

    public AiModel() {
    }

    public AiModel(byte[] model, byte[] tokens) {
        this.model = model;
        this.tokens = tokens;
    }

    public static Builder builder() {
        return new Builder();
    }

    public byte[] getModel() { return model; }
    public void setModel(byte[] model) { this.model = model; }
    public byte[] getTokens() { return tokens; }
    public void setTokens(byte[] tokens) { this.tokens = tokens; }

    public static final class Builder {
        private byte[] model;
        private byte[] tokens;

        private Builder() {}

        public Builder model(byte[] model) { this.model = model; return this; }
        public Builder tokens(byte[] tokens) { this.tokens = tokens; return this; }

        public AiModel build() {
            return new AiModel(model, tokens);
        }
    }
}
