package com.tsvetkov;

import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;

import static com.tsvetkov.RAGUtils.getResourcePath;
import static com.tsvetkov.RAGUtils.initContentRetriever;
import static com.tsvetkov.RAGUtils.initEmbeddingModel;
import static com.tsvetkov.RAGUtils.initOllamaChatModel;
import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

public class RAG {
    public static final String DOCUMENT_FILE = "FluffnookKnowledgeBase.txt";

    public static void main(String[] args) {
        var document = loadDocument( getResourcePath( DOCUMENT_FILE ),  new TextDocumentParser() );

        var splitter = DocumentSplitters.recursive( 400, 50 );
        var segments = splitter.split( document );

        var embeddingModel = initEmbeddingModel();

        List<Embedding> embeddings = embeddingModel.embedAll( segments ).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll( embeddings, segments );

        var contentRetriever = initContentRetriever( embeddingStore, embeddingModel );

        var chatModel = initOllamaChatModel();

        var assistant = AiServices.builder(RagDocumentAssistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
                .build();

        try (var scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("\nEnter your question (or 'exit' to quit): ");
                var question = scanner.nextLine();

                if ("exit".equalsIgnoreCase(question)) {
                    break;
                }

                System.out.println("Thinking...");
                var answer = assistant.answer(question);
                System.out.println("Answer: " + answer);
            }
        }
        System.out.println("Goodbye!");
    }
}