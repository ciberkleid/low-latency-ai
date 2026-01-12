package com.example.low_latency_ai.engine.controller;

import com.example.low_latency_ai.engine.domains.Sentiment;
import com.example.low_latency_ai.engine.service.InferenceService;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@Slf4j
@RequestMapping("ai/inference")
class InferenceRestController {

    private final InferenceService service;

    @SneakyThrows
    @PostMapping("checkSentiment")
    public Sentiment checkSentiment(@RequestBody String text) {
        return service.execute(text);
    }
}
