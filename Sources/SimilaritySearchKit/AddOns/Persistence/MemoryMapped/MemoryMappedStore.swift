//
//  MemoryMappedStore.swift
//
//
//  Created by Sachin Desai on 10/27/25.
//

import Foundation
import SimilaritySearchKit

public final class MemoryMappedStore: MemoryMappedVectorStoreProtocol {
    private struct MetadataFile: Codable {
        let version: Int
        let dimension: Int
        let items: [MemoryMappedIndexContext.Entry]
        let embeddingsFileName: String
    }

    private let metadataExtension = "mmmeta"
    private let embeddingsExtension = "mmdat"
    private let fileManager = FileManager.default

    public init() {}

    public func saveIndex(items: [IndexItem], to url: URL, as name: String) throws -> URL {
        let dimension = items.first?.embedding.count ?? 0
        var currentIndex = 0
        var metadataEntries: [MemoryMappedIndexContext.Entry] = []

        let embeddingsURL = url.appendingPathComponent("\(name).\(embeddingsExtension)")
        if fileManager.fileExists(atPath: embeddingsURL.path) {
            try fileManager.removeItem(at: embeddingsURL)
        }

        fileManager.createFile(atPath: embeddingsURL.path, contents: nil, attributes: nil)
        let handle = try FileHandle(forWritingTo: embeddingsURL)
        defer {
            try? handle.close()
        }

        for item in items {
            if dimension > 0 && item.embedding.count != dimension {
                throw MemoryMappedStoreError.dimensionMismatch(
                    expected: dimension, actual: item.embedding.count)
            }

            let entry = MemoryMappedIndexContext.Entry(
                id: item.id,
                text: item.text,
                metadata: item.metadata,
                embeddingIndex: currentIndex
            )
            metadataEntries.append(entry)

            if !item.embedding.isEmpty {
                let data = item.embedding.withUnsafeBufferPointer { buffer -> Data in
                    Data(
                        bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride
                    )
                }
                try handle.write(contentsOf: data)
                currentIndex += item.embedding.count
            }
        }

        let metadata = MetadataFile(
            version: 1,
            dimension: dimension,
            items: metadataEntries,
            embeddingsFileName: embeddingsURL.lastPathComponent
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let metadataData = try encoder.encode(metadata)

        let metadataURL = url.appendingPathComponent("\(name).\(metadataExtension)")
        try metadataData.write(to: metadataURL, options: .atomic)

        return metadataURL
    }

    public func loadIndex(from url: URL) throws -> [IndexItem] {
        let context = try loadContext(from: url)
        return context.makeIndexItems()
    }

    public func listIndexes(at url: URL) -> [URL] {
        guard
            let files = try? fileManager.contentsOfDirectory(
                at: url, includingPropertiesForKeys: nil)
        else {
            return []
        }
        return files.filter { $0.pathExtension == metadataExtension }
    }

    public func loadContext(from url: URL) throws -> MemoryMappedIndexContext {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let metadata = try decoder.decode(MetadataFile.self, from: data)

        let embeddingsURL = url.deletingLastPathComponent().appendingPathComponent(
            metadata.embeddingsFileName)
        let embeddingsData = try Data(contentsOf: embeddingsURL, options: .mappedIfSafe)

        return MemoryMappedIndexContext(
            entries: metadata.items,
            dimension: metadata.dimension,
            version: metadata.version,
            embeddingsData: embeddingsData
        )
    }
}

public enum MemoryMappedStoreError: Error {
    case dimensionMismatch(expected: Int, actual: Int)
}
