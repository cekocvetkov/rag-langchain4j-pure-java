package com.tsvetkov;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.net.URISyntaxException;
import java.nio.file.Path;

public class RAGUtils {
    static EmbeddingModel initEmbeddingModel()
    {
        return OllamaEmbeddingModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("nomic-embed-text:latest")
                .build();
    }

    static ChatModel initOllamaChatModel()
    {
        return OllamaChatModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("llama3.2:1b")
                .logRequests(true)
                .build();
    }

    static ContentRetriever initContentRetriever(EmbeddingStore<TextSegment> embeddingStore, EmbeddingModel embeddingModel )
    {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore( embeddingStore )
                .embeddingModel( embeddingModel )
                .maxResults(3)
                .minScore(0.5)
                .build();
    }

    static Path getResourcePath(String resourceName) {
        var resourceUrl = RAGUtils.class.getClassLoader().getResource(resourceName);
        if (resourceUrl == null) {
            throw new IllegalArgumentException("Resource not found: " + resourceName);
        }

        try {
            return Path.of(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new IllegalArgumentException("Invalid resource URL: " + resourceName, e);
        }
    }
}
