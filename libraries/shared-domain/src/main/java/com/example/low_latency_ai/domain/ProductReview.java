package com.example.low_latency_ai.domain;

import org.springframework.data.gemfire.mapping.annotation.Region;

@Region("ProductReviews")
public class ProductReview {
    private String id;
    private String productName;
    private String review;

    public ProductReview() {
    }

    public ProductReview(String id, String productName, String review) {
        this.id = id;
        this.productName = productName;
        this.review = review;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public String getReview() { return review; }
    public void setReview(String review) { this.review = review; }

    public static final class Builder {
        private String id;
        private String productName;
        private String review;

        private Builder() {}

        public Builder id(String id) { this.id = id; return this; }
        public Builder productName(String productName) { this.productName = productName; return this; }
        public Builder review(String review) { this.review = review; return this; }

        public ProductReview build() {
            return new ProductReview(id, productName, review);
        }
    }
}
