// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */
import PackageDescription
@usableFromInline
let package = Package(
    name: "TensorKit",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "TensorKit",
            type: .static,
            targets: ["TensorKit"]),
        .library(
            name: "cxxLibrary",
            targets: ["cxxLibrary"])
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "TensorKit",
            dependencies: ["cxxLibrary"],
            swiftSettings: [
                            // Apply optimization settings for release builds
                            .unsafeFlags(["-O"], .when(configuration: .release)),
                            // Apply no optimizations for debug builds
                            .unsafeFlags(["-O"], .when(configuration: .debug)),
                            .interoperabilityMode(.Cxx)
                        ]
        ),
        .target(
            name: "cxxLibrary"
        ),
            
        .testTarget(
            name: "TensorKitTests",
            dependencies: ["TensorKit"],
            swiftSettings: [.interoperabilityMode(.Cxx)]),
    ]
)
