//
//  MXBAIEmbeddings.swift
//
//
//  Created by Sachin Desai on 10/27/25.
//

import CoreML
import Foundation
import SimilaritySearchKit

@available(macOS 13.0, iOS 16.0, *)
public class MXBAIEmbeddings: EmbeddingsProtocol {
    public var supportsBatchEncoding: Bool { true }

    public func encode(sentence: [String]) async -> [[Float]]? {
        guard !sentence.isEmpty else {
            return []
        }

        let batchInputs = sentence.map { makeModelInput(for: $0) }

        if let outputs = try? model.predictions(inputs: batchInputs) {
            var embeddings: [[Float]] = []
            embeddings.reserveCapacity(outputs.count)
            for output in outputs {
                embeddings.append(Self.extractEmbedding(from: output.pooler_output))
            }
            return embeddings
        }

        var fallbackEmbeddings: [[Float]] = []
        fallbackEmbeddings.reserveCapacity(batchInputs.count)
        for input in batchInputs {
            guard let output = try? model.prediction(input: input) else {
                return nil
            }
            fallbackEmbeddings.append(Self.extractEmbedding(from: output.pooler_output))
        }

        return fallbackEmbeddings
    }

    public let model: MXBAI
    public let tokenizer: BertTokenizer
    public let inputDimention: Int = 512
    public let outputDimention: Int = 384

    public init() {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = .cpuAndNeuralEngine

        do {
            self.model = try MXBAI(configuration: modelConfig)
        } catch {
            fatalError("Failed to load the Core ML model. Error: \(error.localizedDescription)")
        }

        self.tokenizer = BertTokenizer()
    }

    // MARK: - Dense Embeddings

    public func encode(sentence: String) async -> [Float]? {
        // Encode input text as bert tokens
        let inputTokens = tokenizer.buildModelTokens(sentence: sentence)
        let (inputIds, attentionMask) = tokenizer.buildModelInputs(from: inputTokens)

        // Send tokens through the MLModel
        let embeddings = generateMXBAIEmbeddings(inputIds: inputIds, attentionMask: attentionMask)

        return embeddings
    }

    public func generateMXBAIEmbeddings(inputIds: MLMultiArray, attentionMask: MLMultiArray)
        -> [Float]?
    {
        let inputFeatures = MXBAIInput(
            input_ids: inputIds,
            attention_mask: attentionMask
        )

        let output = try? model.prediction(input: inputFeatures)

        guard let embeddings = output?.pooler_output else {
            return nil
        }

        let embeddingsArray: [Float] = (0 ..< embeddings.count).map {
            Float(embeddings[$0].floatValue)
        }
        return embeddingsArray
    }

    private func makeModelInput(for text: String) -> MXBAIInput {
        let inputTokens = tokenizer.buildModelTokens(sentence: text)
        let (inputIds, attentionMask) = tokenizer.buildModelInputs(from: inputTokens)
        return MXBAIInput(
            input_ids: inputIds,
            attention_mask: attentionMask
        )
    }

    private static func extractEmbedding(from multiArray: MLMultiArray) -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(multiArray.count)
        for index in 0 ..< multiArray.count {
            result.append(multiArray[index].floatValue)
        }
        return result
    }
}
