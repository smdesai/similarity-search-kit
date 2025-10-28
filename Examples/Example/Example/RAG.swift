//
//  RAG.swift
//  SimilaritySearchKit Example
//
//  Created by Sachin Desai on 10/27/25.
//

import Foundation
import SimilaritySearchKit
import SimilaritySearchKitDistilbert
import SimilaritySearchKitMXBAI
import SimilaritySearchKitMemoryMapped
import SimilaritySearchKitMiniLMAll
import SimilaritySearchKitMiniLMMultiQA

private var ragSimilarityIndex: SimilarityIndex?
private var ragEmbeddingModel: any EmbeddingsProtocol = MXBAIEmbeddings()
private var ragDistanceMetric: any DistanceMetricProtocol = DotProduct()
private var ragVectorStore: any VectorStoreProtocol = JsonStore()
private var currentTokenizer: any TokenizerProtocol = BertTokenizer()
private var currentSplitter: any TextSplitterProtocol = TokenSplitter(
    withTokenizer: BertTokenizer())

public func setCurrentSimilarityIndex(_ index: SimilarityIndex?) {
    ragSimilarityIndex = index
}

public func getCurrentSimilarityIndex() -> SimilarityIndex? {
    ragSimilarityIndex
}

public func configureVectorStore(_ type: VectorStoreType) {
    switch type {
    case .json:
        ragVectorStore = JsonStore()
    case .memoryMapped:
        ragVectorStore = MemoryMappedStore()
    }
}

public func updateIndexComponents(
    currentModel: EmbeddingModelType,
    comparisonAlgorithm: SimilarityMetricType,
    chunkMethod: TextSplitterType
) {
    switch currentModel {
    case .distilbert:
        ragEmbeddingModel = DistilbertEmbeddings()
    case .minilmAll:
        ragEmbeddingModel = MiniLMEmbeddings()
    case .minilmMultiQA:
        ragEmbeddingModel = MultiQAMiniLMEmbeddings()
    case .MXBAI:
        ragEmbeddingModel = MXBAIEmbeddings()
    case .native:
        ragEmbeddingModel = NativeContextualEmbeddings()
    }

    switch comparisonAlgorithm {
    case .dotproduct:
        ragDistanceMetric = DotProduct()
    case .cosine:
        ragDistanceMetric = CosineSimilarity()
    case .euclidian:
        ragDistanceMetric = EuclideanDistance()
    }

    switch chunkMethod {
    case .token:
        currentSplitter = TokenSplitter(withTokenizer: currentTokenizer)
    case .character:
        currentSplitter = CharacterSplitter(withSeparator: " ")
    case .recursive:
        currentSplitter = RecursiveTokenSplitter(withTokenizer: currentTokenizer)
        break
    }
}

public func loadExistingIndex(url: URL, name: String) async {
    let index = await SimilarityIndex(
        name: name,
        model: ragEmbeddingModel,
        metric: ragDistanceMetric,
        vectorStore: ragVectorStore
    )

    do {
        _ = try index.loadIndex(fromDirectory: url, name: name)
        ragSimilarityIndex = index
    } catch {
        print("Failed to load index '\(name)' from \(url.path): \(error)")
    }
}

public func saveIndex(url: URL, name: String) {
    guard let index = ragSimilarityIndex else { return }

    do {
        _ = try index.saveIndex(toDirectory: url, name: name)
    } catch {
        print("Failed to save index '\(name)' to \(url.path): \(error)")
    }
}
