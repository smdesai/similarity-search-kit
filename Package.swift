// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SimilaritySearchKit",
    platforms: [
        .macOS(.v11),
        .iOS(.v15),
    ],
    products: [
        .library(
            name: "SimilaritySearchKit",
            targets: ["SimilaritySearchKit"]
        ),
        .library(
            name: "SimilaritySearchKitMemoryMapped",
            targets: ["SimilaritySearchKitMemoryMapped"]
        ),
        .library(
            name: "SimilaritySearchKitDistilbert",
            targets: ["SimilaritySearchKitDistilbert"]
        ),
        .library(
            name: "SimilaritySearchKitMiniLMAll",
            targets: ["SimilaritySearchKitMiniLMAll"]
        ),
        .library(
            name: "SimilaritySearchKitMiniLMMultiQA",
            targets: ["SimilaritySearchKitMiniLMMultiQA"]
        ),
        .library(
            name: "SimilaritySearchKitMXBAI",
            targets: ["SimilaritySearchKitMXBAI"]
        ),
    ],
    targets: [
        .target(
            name: "SimilaritySearchKit",
            dependencies: [],
            path: "Sources/SimilaritySearchKit/Core",
            resources: [.process("Resources")]
        ),
        .target(
            name: "SimilaritySearchKitMemoryMapped",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Persistence/MemoryMapped"
        ),
        .target(
            name: "SimilaritySearchKitDistilbert",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Embeddings/Distilbert"
        ),
        .target(
            name: "SimilaritySearchKitMiniLMAll",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Embeddings/MiniLMAll"
        ),
        .target(
            name: "SimilaritySearchKitMiniLMMultiQA",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Embeddings/MiniLMMultiQA"
        ),
        .target(
            name: "SimilaritySearchKitMXBAI",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Embeddings/MXBAI"
        ),
        .testTarget(
            name: "SimilaritySearchKitTests",
            dependencies: [
                "SimilaritySearchKit",
                "SimilaritySearchKitDistilbert",
                "SimilaritySearchKitMiniLMAll",
                "SimilaritySearchKitMiniLMMultiQA",
                "SimilaritySearchKitMXBAI",
            ],
            path: "Tests/SimilaritySearchKitTests",
            resources: [.process("Resources")]
        ),
    ]
)
