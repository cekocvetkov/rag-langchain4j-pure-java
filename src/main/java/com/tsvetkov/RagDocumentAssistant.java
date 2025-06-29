package com.tsvetkov;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface RagDocumentAssistant {
    @SystemMessage("You are a helpful assistant that answers questions about a fictional animal called Fluffnook.")
    String answer(@UserMessage String question);
}