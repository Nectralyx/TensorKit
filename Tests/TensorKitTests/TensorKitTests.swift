import XCTest
@testable import TensorKit

final class TensorKitTests: XCTestCase {
    func testExample() throws {
        // XCTest Documentation
        // https://developer.apple.com/documentation/xctest
        let tt1 = CFAbsoluteTimeGetCurrent()
        let t4 = TensorKit.Tensor<Float>(.mean_scaling, shape: [1, 1, 100])
        let tt2 = CFAbsoluteTimeGetCurrent()
        let t5 = TensorKit.Tensor<Float>(.he, shape: [1000, 1000, 100])
        let tt3 = CFAbsoluteTimeGetCurrent()
        let t6 = t4 * t5
        let tt4 = CFAbsoluteTimeGetCurrent()
        print("B: \(tt2 - tt1) : \(tt3 - tt2) : \(tt4 - tt3)")
        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods
    }
}
