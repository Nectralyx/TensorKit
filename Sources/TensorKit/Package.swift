// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.
/*
 * Copyright (c) 2024 Nectralyx.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0.
 *
 * SPDX-License-Identifier: EPL-2.0
 */

import PackageDescription

let package = Package(
    name: "TensorKit",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "TensorKit",
            targets: ["TensorKit"]),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        //.binaryTarget(name: "TensorKitFramework", path: "TensorKitFramework.xcframework"),
        .target(
            name: "TensorKit"),
        .testTarget(
            name: "TensorKitTests",
            dependencies: ["TensorKit"]),
    ]
)
