//
//  ContentView.swift
//  PDFExample
//
//  Created by Zach Nagengast on 5/2/23.
//

import PDFKit
import SimilaritySearchKit
import SimilaritySearchKitDistilbert
import SimilaritySearchKitMXBAI
import SimilaritySearchKitMemoryMapped
import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var documentText: String = ""
    @State private var fileName: String = ""
    @State private var fileIcon: UIImage? = nil
    @State private var totalCharacters: Int = 0
    @State private var totalTokens: Int = 0
    @State private var chunks: [String] = []
    @State private var embeddings: [[Float]] = []
    @State private var searchText: String = ""
    @State private var searchResults: [(text: String, source: String)] = []
    @State private var isVectorizing: Bool = false
    @State private var vectorizingProgress: Double = 0.0
    @State private var indexedFileName: String = ""
    @State private var embeddingTime: TimeInterval = 0.0
    @State private var lastQueryTime: TimeInterval = 0.0
    @State private var isBatchProcessing: Bool = false
    @State private var batchFileProgress: Double = 0.0
    @State private var batchCurrentFile: String = ""
    @State private var batchTotalFiles: Int = 0
    @State private var batchCurrentFileIndex: Int = 0

    @State private var similarityIndex: SimilarityIndex?
    @State private var ragDirectoryURL: URL? = nil

    private let ragIndexName = "RAGExampleIndex"

    private var canSearch: Bool {
        // Can search if: we have a file loaded and it's been indexed, OR we have batch indexed files
        if !fileName.isEmpty {
            // If we have a specific file loaded, check if it's indexed
            return indexedFileName == fileName && currentEmbeddingCount > 0
        } else {
            // If no specific file is loaded, check if we have any indexed content
            return totalIndexedEmbeddings > 0
        }
    }

    private var currentEmbeddingCount: Int {
        // Only count embeddings for the currently indexed file
        guard !fileName.isEmpty, indexedFileName == fileName else { return 0 }

        if !embeddings.isEmpty {
            return embeddings.count
        }

        guard let index = similarityIndex else { return 0 }

        // Count items matching the current indexed file
        let matchingItems = index.indexItems.filter { $0.metadata["source"] == indexedFileName }
        if !matchingItems.isEmpty {
            return matchingItems.count
        }

        // Check memory-mapped entries for the current file
        let memoryMappedItems = index.memoryMappedItems { entry in
            entry.metadata["source"] == indexedFileName
        }
        return memoryMappedItems.count
    }

    private var totalIndexedEmbeddings: Int {
        guard let index = similarityIndex else { return 0 }

        // Prefer in-memory items count
        let inMemoryCount = index.indexItems.count
        if inMemoryCount > 0 {
            return inMemoryCount
        }

        // Fall back to memory-mapped count
        return index.memoryMappedEntries().count
    }

    private var needsIndexing: Bool {
        return !fileName.isEmpty && indexedFileName != fileName
    }

    private func ensureRagDirectory() -> URL? {
        guard
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
                .first
        else {
            print("Unable to locate documents directory.")
            return nil
        }

        let ragDirectory = documents.appendingPathComponent("RAGIndexStore", isDirectory: true)
        if !FileManager.default.fileExists(atPath: ragDirectory.path) {
            do {
                try FileManager.default.createDirectory(
                    at: ragDirectory, withIntermediateDirectories: true)
            } catch {
                print("Failed to create RAG directory:", error)
                return nil
            }
        }

        return ragDirectory
    }

    private func indexHasContent(_ index: SimilarityIndex) -> Bool {
        if !index.indexItems.isEmpty {
            return true
        }

        return !index.memoryMappedEntries().isEmpty
    }

    private func filteredItems(from index: SimilarityIndex, matching fileName: String?)
        -> [SimilarityIndex.IndexItem]
    {
        guard let fileName = fileName, !fileName.isEmpty else {
            return []
        }

        if !index.indexItems.isEmpty {
            return index.indexItems.filter { $0.metadata["source"] == fileName }
        }

        return index.memoryMappedItems { entry in
            entry.metadata["source"] == fileName
        }
    }

    private func makeStateSnapshot(
        from items: [SimilarityIndex.IndexItem], defaultFileName: String?
    ) -> (
        document: String, tokens: Int, characters: Int, chunks: [String], embeddings: [[Float]],
        fileName: String
    ) {
        let combinedText = items.map(\.text).joined(separator: "\n\n")
        let tokens = BertTokenizer().tokenize(text: combinedText).count
        let resolvedName = defaultFileName ?? items.first?.metadata["source"] ?? ""
        return (
            combinedText, tokens, combinedText.count, items.map(\.text), items.map(\.embedding),
            resolvedName
        )
    }

    private func loadStoredIndex(filteringBy fileName: String? = nil) async -> (
        SimilarityIndex, [SimilarityIndex.IndexItem]
    )? {
        guard let directory = ragDirectoryURL ?? ensureRagDirectory() else {
            return nil
        }

        await MainActor.run {
            ragDirectoryURL = directory
        }

        configureVectorStore(.memoryMapped)
        updateIndexComponents(
            currentModel: .MXBAI, comparisonAlgorithm: .dotproduct, chunkMethod: .recursive)
        await loadExistingIndex(url: directory, name: ragIndexName)

        guard let index = getCurrentSimilarityIndex() else {
            return nil
        }

        let items = filteredItems(from: index, matching: fileName)
        await MainActor.run {
            if !items.isEmpty {
                index.indexItems = items
            }
            similarityIndex = index
        }
        setCurrentSimilarityIndex(index)
        return (index, items)
    }

    private func ensureSimilarityIndex() async -> SimilarityIndex {
        if let index = similarityIndex {
            return index
        }

        configureVectorStore(.memoryMapped)
        let newIndex = await SimilarityIndex(
            name: "RAGIndex",
            model: MXBAIEmbeddings(),
            metric: DotProduct(),
            vectorStore: MemoryMappedStore()
        )

        await MainActor.run {
            similarityIndex = newIndex
            setCurrentSimilarityIndex(newIndex)
        }

        return newIndex
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Compact File Info Card
                    if !fileName.isEmpty {
                        VStack(spacing: 12) {
                            HStack(spacing: 12) {
                                if let fileIcon = fileIcon {
                                    Image(uiImage: fileIcon)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(width: 40, height: 40)
                                        .cornerRadius(6)
                                }

                                VStack(alignment: .leading, spacing: 4) {
                                    Text(fileName)
                                        .font(.subheadline)
                                        .fontWeight(.semibold)
                                        .lineLimit(1)
                                    Text("\(totalTokens) tokens")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    if currentEmbeddingCount > 0 {
                                        if embeddingTime > 0 && indexedFileName == fileName {
                                            Text(
                                                "\(currentEmbeddingCount) embeddings · \(String(format: "%.2f", embeddingTime))s"
                                            )
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                        } else {
                                            Text("\(currentEmbeddingCount) embeddings")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                }

                                Spacer()
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(10)
                        }
                        .frame(maxWidth: 500)
                        .padding(.horizontal)
                    }

                    // Vectorizing Progress Indicator
                    if isVectorizing && !isBatchProcessing {
                        VStack(spacing: 8) {
                            ProgressView(value: vectorizingProgress, total: 1.0)
                                .progressViewStyle(LinearProgressViewStyle())
                                .frame(maxWidth: 500)
                            Text("Creating embeddings... \(Int(vectorizingProgress * 100))%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                    }

                    // Batch Processing Progress Indicators
                    if isBatchProcessing {
                        VStack(spacing: 16) {
                            // Overall file progress
                            VStack(spacing: 8) {
                                HStack {
                                    Text("Processing files")
                                        .font(.subheadline)
                                        .fontWeight(.semibold)
                                    Spacer()
                                    Text("\(batchCurrentFileIndex) of \(batchTotalFiles)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                ProgressView(value: batchFileProgress, total: 1.0)
                                    .progressViewStyle(LinearProgressViewStyle())
                            }
                            .frame(maxWidth: 500)

                            // Current file embedding progress
                            if !batchCurrentFile.isEmpty {
                                VStack(spacing: 8) {
                                    HStack {
                                        Text(batchCurrentFile)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                            .lineLimit(1)
                                        Spacer()
                                        Text("\(Int(vectorizingProgress * 100))%")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    ProgressView(value: vectorizingProgress, total: 1.0)
                                        .progressViewStyle(LinearProgressViewStyle())
                                }
                                .frame(maxWidth: 500)
                            }
                        }
                        .padding()
                    }

                    // Search Section
                    if canSearch && !isVectorizing && !isBatchProcessing {
                        VStack(spacing: 16) {
                            VStack(spacing: 8) {
                                HStack {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                    Text("\(totalIndexedEmbeddings) embeddings ready")
                                        .font(.subheadline)
                                        .foregroundColor(.secondary)
                                }

                                if embeddingTime > 0 && !fileName.isEmpty
                                    && indexedFileName == fileName
                                {
                                    Text(
                                        "Embedding time: \(String(format: "%.2f", embeddingTime))s"
                                    )
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                }
                            }

                            HStack {
                                Image(systemName: "magnifyingglass")
                                    .foregroundColor(.secondary)
                                TextField(
                                    "Search document", text: $searchText, onCommit: searchDocument)
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(10)
                            .frame(maxWidth: 500)

                            // Search Results
                            if !searchResults.isEmpty {
                                VStack(alignment: .leading, spacing: 12) {
                                    HStack {
                                        Text("Search Results")
                                            .font(.headline)
                                        Spacer()
                                        if lastQueryTime > 0 {
                                            Text("\(String(format: "%.3f", lastQueryTime))s")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                    .padding(.horizontal)

                                    ForEach(Array(searchResults.enumerated()), id: \.offset) {
                                        index, result in
                                        VStack(alignment: .leading, spacing: 8) {
                                            HStack {
                                                Text("Result \(index + 1)")
                                                    .font(.caption)
                                                    .fontWeight(.semibold)
                                                    .foregroundColor(.blue)
                                                Spacer()
                                                Text(result.source)
                                                    .font(.caption)
                                                    .foregroundColor(.secondary)
                                                    .lineLimit(1)
                                            }
                                            Text(result.text)
                                                .font(.body)
                                                .foregroundColor(.primary)
                                        }
                                        .padding()
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .background(Color(.systemGray6))
                                        .cornerRadius(10)
                                    }
                                }
                                .frame(maxWidth: 500)
                                .padding(.horizontal)
                            }
                        }
                        .padding(.top)
                    }

                    Spacer()
                }
            }
            .navigationTitle("Search")
            .toolbar {
                ToolbarItemGroup(placement: .navigationBarLeading) {
                    Button(action: selectFromFiles) {
                        Label("Add File", systemImage: "plus")
                    }
                    Button(action: batchLoadFiles) {
                        Label("Batch Load", systemImage: "folder.badge.plus")
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: clearAllIndexes) {
                        Label("Clear Indexes", systemImage: "trash")
                    }
                    .foregroundColor(.red)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
        }
        .onAppear {
            loadIndex()
        }
    }

    func loadIndex() {
        Task {
            configureVectorStore(.memoryMapped)
            updateIndexComponents(
                currentModel: .MXBAI, comparisonAlgorithm: .dotproduct, chunkMethod: .recursive)

            if let directory = ragDirectoryURL ?? ensureRagDirectory() {
                await MainActor.run {
                    ragDirectoryURL = directory
                }

                await loadExistingIndex(url: directory, name: ragIndexName)

                if let existingIndex = getCurrentSimilarityIndex() {
                    // Check for dimension mismatch - if dimensions don't match, clear the old index
                    let newIndex = await ensureSimilarityIndex()
                    if existingIndex.dimension != 0 && newIndex.dimension != 0
                        && existingIndex.dimension != newIndex.dimension
                    {
                        print(
                            "⚠️ Dimension mismatch detected! Old: \(existingIndex.dimension), New: \(newIndex.dimension)"
                        )
                        print("Clearing old index to prevent compatibility issues...")

                        // Clear the old index files
                        do {
                            let fileURLs = try FileManager.default.contentsOfDirectory(
                                at: directory,
                                includingPropertiesForKeys: nil,
                                options: .skipsHiddenFiles
                            )
                            for fileURL in fileURLs {
                                try FileManager.default.removeItem(at: fileURL)
                            }
                            print("✓ Old index cleared. Ready for new embeddings.")
                        } catch {
                            print("Error clearing old index: \(error)")
                        }

                        await MainActor.run {
                            similarityIndex = newIndex
                            setCurrentSimilarityIndex(newIndex)
                        }
                        return
                    }

                    // Don't load memory-mapped items into indexItems - leave them in storage
                    // This avoids memory issues and crashes
                    print(
                        "Index loaded with \(existingIndex.indexItems.count) in-memory items and \(existingIndex.memoryMappedEntries().count) memory-mapped entries"
                    )

                    await MainActor.run {
                        similarityIndex = existingIndex
                        setCurrentSimilarityIndex(existingIndex)
                    }

                    if indexHasContent(existingIndex) {
                        return
                    }
                }
            }

            _ = await ensureSimilarityIndex()
        }
    }

    func selectFromFiles() {
        let picker = DocumentPicker(
            document: $documentText,
            fileName: $fileName,
            fileIcon: $fileIcon,
            totalCharacters: $totalCharacters,
            totalTokens: $totalTokens,
            onFileLoaded: handleLoadedFile(url:extractedText:)
        )
        let hostingController = UIHostingController(rootView: picker)
        UIApplication.shared.connectedScenes
            .map { ($0 as? UIWindowScene)?.windows.first?.rootViewController }
            .compactMap { $0 }
            .first?
            .present(hostingController, animated: true, completion: nil)
    }

    private func handleLoadedFile(url: URL, extractedText: String) {
        Task {
            if let (storedIndex, storedItems) = await loadStoredIndex(
                filteringBy: url.lastPathComponent), !storedItems.isEmpty
            {
                let snapshot = makeStateSnapshot(
                    from: storedItems, defaultFileName: url.lastPathComponent)

                await MainActor.run {
                    storedIndex.indexItems = storedItems
                    similarityIndex = storedIndex
                    documentText = snapshot.document
                    totalCharacters = snapshot.characters
                    totalTokens = snapshot.tokens
                    chunks = snapshot.chunks
                    embeddings = snapshot.embeddings
                    fileName = snapshot.fileName
                    indexedFileName = snapshot.fileName
                    searchResults = []
                    embeddingTime = 0.0
                    lastQueryTime = 0.0
                    setCurrentSimilarityIndex(storedIndex)
                }
            } else {
                let tokens = BertTokenizer().tokenize(text: extractedText).count

                await MainActor.run {
                    documentText = extractedText
                    totalCharacters = extractedText.count
                    totalTokens = tokens
                    chunks = []
                    embeddings = []
                    searchResults = []
                    fileName = url.lastPathComponent
                    indexedFileName = ""
                    embeddingTime = 0.0
                    lastQueryTime = 0.0
                }

                _ = await ensureSimilarityIndex()

                // Automatically start generating embeddings for new files
                await MainActor.run {
                    vectorizeChunks()
                }
            }
        }
    }

    func vectorizeChunks() {
        guard !documentText.isEmpty, !fileName.isEmpty else { return }

        Task {
            let startTime = Date()

            await MainActor.run {
                searchResults = []
                isVectorizing = true
                vectorizingProgress = 0.0
            }

            let index = await ensureSimilarityIndex()
            let splitter = RecursiveTokenSplitter(withTokenizer: BertTokenizer())
            let (splitText, _) = splitter.split(text: documentText)

            let embeddingModel = index.indexModel
            var preparedChunks: [String] = []
            var generatedEmbeddings: [[Float]] = []

            let totalChunks = splitText.count
            print("Generating embeddings for \(totalChunks) chunks...")
            for (idx, chunk) in splitText.enumerated() {
                if let embedding = await embeddingModel.encode(sentence: chunk) {
                    preparedChunks.append(chunk)
                    generatedEmbeddings.append(embedding)

                    // Update progress every 10 chunks or on the last chunk to reduce overhead
                    if (idx + 1) % 10 == 0 || idx == totalChunks - 1 {
                        await MainActor.run {
                            vectorizingProgress = Double(idx + 1) / Double(totalChunks)
                        }
                    }
                }
            }
            let embeddingEndTime = Date()
            let embeddingDuration = embeddingEndTime.timeIntervalSince(startTime)
            print(
                "✓ Embedding generation completed in \(String(format: "%.2f", embeddingDuration))s")

            print("Adding \(preparedChunks.count) items to index...")
            let addItemsStartTime = Date()

            // Remove existing items for this file to avoid conflicts
            let existingItemIDs = index.indexItems
                .filter { $0.metadata["source"] == fileName }
                .map(\.id)
            existingItemIDs.forEach { index.removeItem(id: $0) }

            let ids = preparedChunks.enumerated().map { "\(fileName)-chunk-\($0.offset)" }
            let metadataEntries = Array(
                repeating: ["source": fileName], count: preparedChunks.count)
            let embeddingPayload = generatedEmbeddings.map { Optional($0) }
            await index.addItems(
                ids: ids,
                texts: preparedChunks,
                metadata: metadataEntries,
                embeddings: embeddingPayload
            )

            let addItemsEndTime = Date()
            let addItemsTime = addItemsEndTime.timeIntervalSince(addItemsStartTime)
            print("✓ Added items to index in \(String(format: "%.2f", addItemsTime))s")

            setCurrentSimilarityIndex(index)
            await MainActor.run {
                similarityIndex = index
            }

            print("Saving index to disk...")
            let saveStartTime = Date()
            if let directory = ragDirectoryURL ?? ensureRagDirectory() {
                await MainActor.run {
                    ragDirectoryURL = directory
                }
                saveIndex(url: directory, name: ragIndexName)
            }
            let saveEndTime = Date()
            let saveTime = saveEndTime.timeIntervalSince(saveStartTime)
            print("✓ Index saved to disk in \(String(format: "%.2f", saveTime))s")

            let endTime = Date()
            let elapsedTime = endTime.timeIntervalSince(startTime)
            print(
                "=== Total time: \(String(format: "%.2f", elapsedTime))s (Embedding: \(String(format: "%.2f", embeddingDuration))s, Adding: \(String(format: "%.2f", addItemsTime))s, Saving: \(String(format: "%.2f", saveTime))s) ==="
            )

            await MainActor.run {
                chunks = preparedChunks
                embeddings = generatedEmbeddings
                indexedFileName = fileName
                embeddingTime = embeddingDuration
                isVectorizing = false
                vectorizingProgress = 0.0
            }
        }
    }

    func searchDocument() {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return }

        Task {
            let startTime = Date()

            var activeIndex = similarityIndex

            // If we don't have an index or it's empty, try to load from disk
            if activeIndex == nil || !(activeIndex.map(indexHasContent) ?? false) {
                print("Loading index from disk for search...")
                if let (storedIndex, _) = await loadStoredIndex(filteringBy: nil) {
                    activeIndex = storedIndex
                    await MainActor.run {
                        similarityIndex = storedIndex
                    }
                }
            }

            guard let index = activeIndex else {
                print("No index available for search")
                return
            }

            guard indexHasContent(index) else {
                print(
                    "Index has no content. Items: \(index.indexItems.count), MemMapped: \(index.memoryMappedEntries().count)"
                )
                return
            }

            let itemCount = index.indexItems.count
            let memMappedCount = index.memoryMappedEntries().count
            print(
                "Searching with query: '\(query)' in index with \(itemCount) items and \(memMappedCount) memory-mapped entries"
            )

            if itemCount == 0 && memMappedCount > 0 {
                print(
                    "Warning: Only memory-mapped entries exist. Loading items from memory-mapped storage..."
                )
                // Load all items from memory-mapped storage into the index
                let allItems = index.memoryMappedItems { _ in true }
                print("Loaded \(allItems.count) items from memory-mapped storage")

                // Populate the index's indexItems array so search can work
                index.indexItems = allItems
                print("Index now has \(index.indexItems.count) searchable items")

                // Update the stored index
                setCurrentSimilarityIndex(index)
                await MainActor.run {
                    similarityIndex = index
                }
            }

            let results = await index.search(query)
            print("Search returned \(results.count) results")

            let endTime = Date()
            let elapsedTime = endTime.timeIntervalSince(startTime)

            await MainActor.run {
                searchResults = results.map { result in
                    let source = result.metadata["source"] as? String ?? "Unknown"
                    return (text: result.text, source: source)
                }
                lastQueryTime = elapsedTime
            }
        }
    }

    func clearAllIndexes() {
        Task {
            guard let directory = ragDirectoryURL ?? ensureRagDirectory() else { return }

            do {
                // Get all files in the directory
                let fileURLs = try FileManager.default.contentsOfDirectory(
                    at: directory,
                    includingPropertiesForKeys: nil,
                    options: .skipsHiddenFiles
                )

                // Remove all files
                for fileURL in fileURLs {
                    try FileManager.default.removeItem(at: fileURL)
                }

                // Reset state
                await MainActor.run {
                    similarityIndex = nil
                    documentText = ""
                    fileName = ""
                    fileIcon = nil
                    totalCharacters = 0
                    totalTokens = 0
                    chunks = []
                    embeddings = []
                    searchText = ""
                    searchResults = []
                    indexedFileName = ""
                    embeddingTime = 0.0
                    lastQueryTime = 0.0
                    isVectorizing = false
                    vectorizingProgress = 0.0
                }

                // Create a new index
                _ = await ensureSimilarityIndex()

                print("All indexes cleared successfully")
            } catch {
                print("Error clearing indexes: \(error)")
            }
        }
    }

    func batchLoadFiles() {
        Task {
            guard
                let documentsDirectory = FileManager.default.urls(
                    for: .documentDirectory, in: .userDomainMask
                ).first
            else {
                print("Could not access documents directory")
                return
            }

            let filesDirectory = documentsDirectory.appendingPathComponent(
                "Files", isDirectory: true)

            // Check if Files directory exists
            guard FileManager.default.fileExists(atPath: filesDirectory.path) else {
                print("Files directory does not exist at \(filesDirectory.path)")
                return
            }

            do {
                // Get all files in the Files directory
                let fileURLs = try FileManager.default.contentsOfDirectory(
                    at: filesDirectory,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: .skipsHiddenFiles
                ).filter { url in
                    let ext = url.pathExtension.lowercased()
                    return ext == "pdf" || ext == "txt"
                }

                guard !fileURLs.isEmpty else {
                    print("No PDF or TXT files found in Files directory")
                    return
                }

                await MainActor.run {
                    isBatchProcessing = true
                    batchTotalFiles = fileURLs.count
                    batchCurrentFileIndex = 0
                    batchFileProgress = 0.0
                }

                let index = await ensureSimilarityIndex()

                for (fileIndex, fileURL) in fileURLs.enumerated() {
                    let needsStopAccessing = fileURL.startAccessingSecurityScopedResource()
                    let currentFileName = fileURL.lastPathComponent
                    let fileExtension = fileURL.pathExtension.lowercased()

                    await MainActor.run {
                        batchCurrentFile = currentFileName
                        batchCurrentFileIndex = fileIndex + 1
                        batchFileProgress = Double(fileIndex) / Double(fileURLs.count)
                        isVectorizing = true
                        vectorizingProgress = 0.0
                    }

                    var extractedText = ""

                    // Extract text based on file type
                    if fileExtension == "txt" {
                        do {
                            extractedText = try String(contentsOf: fileURL, encoding: .utf8)
                        } catch {
                            print("Failed to read text file \(currentFileName): \(error)")
                            if needsStopAccessing {
                                fileURL.stopAccessingSecurityScopedResource()
                            }
                            continue
                        }
                    } else if fileExtension == "pdf" {
                        guard let document = PDFDocument(url: fileURL) else {
                            print("Failed to open PDF document \(currentFileName)")
                            if needsStopAccessing {
                                fileURL.stopAccessingSecurityScopedResource()
                            }
                            continue
                        }

                        for pageIndex in 0 ..< document.pageCount {
                            if let page = document.page(at: pageIndex),
                                let pageContent = page.string
                            {
                                extractedText += pageContent
                            }
                        }
                    }

                    if needsStopAccessing {
                        fileURL.stopAccessingSecurityScopedResource()
                    }

                    // Skip empty files
                    guard !extractedText.isEmpty else {
                        print("Skipping empty file: \(currentFileName)")
                        continue
                    }

                    // Generate embeddings for this file
                    let splitter = RecursiveTokenSplitter(withTokenizer: BertTokenizer())
                    let (splitText, _) = splitter.split(text: extractedText)

                    let embeddingModel = index.indexModel
                    let totalChunks = splitText.count

                    // Remove existing items for this file
                    let existingItemIDs = index.indexItems
                        .filter { $0.metadata["source"] == currentFileName }
                        .map(\.id)
                    existingItemIDs.forEach { index.removeItem(id: $0) }

                    // Generate embeddings and add items
                    for (idx, chunk) in splitText.enumerated() {
                        if let embedding = await embeddingModel.encode(sentence: chunk) {
                            let identifier = "\(currentFileName)-chunk-\(idx)"
                            await index.addItem(
                                id: identifier, text: chunk, metadata: ["source": currentFileName],
                                embedding: embedding)

                            // Update progress every 10 chunks or on last chunk
                            if (idx + 1) % 10 == 0 || idx == totalChunks - 1 {
                                await MainActor.run {
                                    vectorizingProgress = Double(idx + 1) / Double(totalChunks)
                                }
                            }
                        }
                    }
                }

                // Save the index to disk
                if let directory = ragDirectoryURL ?? ensureRagDirectory() {
                    await MainActor.run {
                        ragDirectoryURL = directory
                    }
                    print("Saving index with \(index.indexItems.count) items to \(directory.path)")
                    saveIndex(url: directory, name: ragIndexName)
                    print("Index saved successfully")

                    // After saving, indexItems may be cleared. Reload them from memory-mapped storage
                    if index.indexItems.isEmpty && !index.memoryMappedEntries().isEmpty {
                        print("Reloading items from memory-mapped storage after save...")
                        let allItems = index.memoryMappedItems { _ in true }
                        index.indexItems = allItems
                        print("Reloaded \(index.indexItems.count) items into memory")
                    }
                }

                // Keep the current index with all items in memory for searching
                setCurrentSimilarityIndex(index)
                await MainActor.run {
                    similarityIndex = index
                    batchFileProgress = 1.0
                    isBatchProcessing = false
                    isVectorizing = false
                    vectorizingProgress = 0.0
                    batchCurrentFile = ""
                }

                print(
                    "Batch processing completed: \(fileURLs.count) files indexed. Index has \(index.indexItems.count) items ready for search"
                )

            } catch {
                print("Error during batch processing: \(error)")
                await MainActor.run {
                    isBatchProcessing = false
                    isVectorizing = false
                }
            }
        }
    }

}

// MARK: - Document Picker

struct DocumentPicker: UIViewControllerRepresentable {
    @Binding var document: String
    @Binding var fileName: String
    @Binding var fileIcon: UIImage?
    @Binding var totalCharacters: Int
    @Binding var totalTokens: Int

    var onFileLoaded: (URL, String) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.pdf, .plainText])
        picker.delegate = context.coordinator
        picker.shouldShowFileExtensions = true
        return picker
    }

    func updateUIViewController(_: UIDocumentPickerViewController, context _: Context) {}

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        var parent: DocumentPicker

        init(_ parent: DocumentPicker) {
            self.parent = parent
        }

        func documentPicker(_: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }

            let needsStopAccessing = url.startAccessingSecurityScopedResource()
            let fileExtension = url.pathExtension.lowercased()

            var extractedText = ""
            var icon: UIImage?

            if fileExtension == "txt" {
                // Handle text files
                do {
                    extractedText = try String(contentsOf: url, encoding: .utf8)
                    icon = UIImage(systemName: "doc.text")
                } catch {
                    if needsStopAccessing {
                        url.stopAccessingSecurityScopedResource()
                    }
                    print("Failed to read text file at \(url): \(error)")
                    return
                }
            } else if fileExtension == "pdf" {
                // Handle PDF files
                guard let document = PDFDocument(url: url) else {
                    if needsStopAccessing {
                        url.stopAccessingSecurityScopedResource()
                    }
                    print("Failed to open PDF document at \(url)")
                    return
                }

                for pageIndex in 0 ..< document.pageCount {
                    if let page = document.page(at: pageIndex),
                        let pageContent = page.string
                    {
                        extractedText += pageContent
                    }
                }

                if let firstPage = document.page(at: 0) {
                    icon = firstPage.thumbnail(of: CGSize(width: 60, height: 60), for: .cropBox)
                }
            } else {
                if needsStopAccessing {
                    url.stopAccessingSecurityScopedResource()
                }
                print("Unsupported file type: \(fileExtension)")
                return
            }

            let tokenCount = BertTokenizer().tokenize(text: extractedText).count

            if needsStopAccessing {
                url.stopAccessingSecurityScopedResource()
            }

            DispatchQueue.main.async {
                self.parent.document = extractedText
                self.parent.fileName = url.lastPathComponent
                self.parent.totalCharacters = extractedText.count
                self.parent.totalTokens = tokenCount
                self.parent.fileIcon = icon
                self.parent.onFileLoaded(url, extractedText)
            }
        }
    }
}

// MARK: - Previews

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
