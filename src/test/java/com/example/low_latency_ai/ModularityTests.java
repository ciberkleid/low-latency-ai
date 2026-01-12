package com.example.low_latency_ai;

import org.junit.jupiter.api.Test;
import org.springframework.modulith.core.ApplicationModules;
import org.springframework.modulith.docs.Documenter;

class ModularityTests {

    ApplicationModules modules = ApplicationModules.of(LowLatencyAiApplication.class);

    @Test
    void verifiesArchitecture() {

        System.out.println(modules);

        modules.verify();
    }

    @Test
    void createDocumentation() {
        new Documenter(modules).writeDocumentation();
    }

}
