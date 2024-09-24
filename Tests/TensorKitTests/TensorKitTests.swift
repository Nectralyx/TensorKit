/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import XCTest
@testable import TensorKit

final class TensorKitTests: XCTestCase {
    func testExample() throws {
        // XCTest Documentation
        // https://developer.apple.com/documentation/xctest
        let tt1 = CFAbsoluteTimeGetCurrent()
        let t4 = TensorKit.Tensor<Float>([[3, 4, 5]])
        let tt2 = CFAbsoluteTimeGetCurrent()
        let t5 = TensorKit.Tensor<Float>(.he, shape: [1000, 1000, 100])
        let tt3 = CFAbsoluteTimeGetCurrent()
        //let t6 = t4 + t5
        let t7 = t4.expand(to: [3, 3, 3])
        let tt4 = CFAbsoluteTimeGetCurrent()
        print("B: \(tt2 - tt1) : \(tt3 - tt2) : \(tt4 - tt3)")
        print(t7)
        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods
    }
}
