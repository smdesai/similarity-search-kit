//
//  MemoryMappedIndexContext.swift
//
//
//  Created by Sachin Desai on 10/27/25.
//

import Foundation

public struct MemoryMappedIndexContext {
    public struct Entry: Codable {
        public let id: String
        public let text: String
        public let metadata: [String: String]
        public let embeddingIndex: Int

        public init(id: String, text: String, metadata: [String: String], embeddingIndex: Int) {
            self.id = id
            self.text = text
            self.metadata = metadata
            self.embeddingIndex = embeddingIndex
        }
    }

    public let entries: [Entry]
    public let dimension: Int
    public let version: Int
    private let embeddingsData: Data

    public init(entries: [Entry], dimension: Int, version: Int, embeddingsData: Data) {
        self.entries = entries
        self.dimension = dimension
        self.version = version
        self.embeddingsData = embeddingsData
    }

    public func makeIndexItems(excluding excludedIDs: Set<String> = []) -> [IndexItem] {
        guard dimension > 0 else {
            return
                entries
                .filter { !excludedIDs.contains($0.id) }
                .map { entry in
                    IndexItem(
                        id: entry.id, text: entry.text, embedding: [], metadata: entry.metadata)
                }
        }

        var result: [IndexItem] = []
        enumerateEmbeddings(excluding: excludedIDs) { entry, buffer in
            let vector = Array(buffer)
            let item = IndexItem(
                id: entry.id, text: entry.text, embedding: vector, metadata: entry.metadata)
            result.append(item)
        }
        return result
    }

    public func enumerateEmbeddings(
        excluding excludedIDs: Set<String> = [],
        _ body: (Entry, UnsafeBufferPointer<Float>) -> Void
    ) {
        guard dimension > 0 else { return }

        embeddingsData.withUnsafeBytes { rawBuffer in
            let floatBuffer = rawBuffer.bindMemory(to: Float.self)
            guard let baseAddress = floatBuffer.baseAddress else { return }

            for entry in entries where !excludedIDs.contains(entry.id) {
                let start = baseAddress.advanced(by: entry.embeddingIndex)
                let pointer = UnsafeBufferPointer(start: start, count: dimension)
                body(entry, pointer)
            }
        }
    }

    public func makeIndexItems(for entries: [Entry]) -> [IndexItem] {
        guard !entries.isEmpty else { return [] }

        if dimension == 0 {
            return entries.map { entry in
                IndexItem(id: entry.id, text: entry.text, embedding: [], metadata: entry.metadata)
            }
        }

        let idSet = Set(entries.map { $0.id })
        var items: [IndexItem] = []
        enumerateEmbeddings { entry, buffer in
            guard idSet.contains(entry.id) else { return }
            let vector = Array(buffer)
            let item = IndexItem(
                id: entry.id, text: entry.text, embedding: vector, metadata: entry.metadata)
            items.append(item)
        }
        return items
    }
}
