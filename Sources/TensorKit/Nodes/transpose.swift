//
//  transpose.swift
//  TensorKit
//
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */
import Foundation
import Accelerate

public extension Tensor {
    @inlinable
    func transpose() -> Tensor {
        let result = Tensor(.empty, shape: shape, calculate_grad: gradient != nil)
        let rows = shape.dropLast().last!
        let cols = shape.last!
        let totalElements = dataSize
        let batchCount = shape.dropLast(2).reduce(1, *)
        result.shape.swapAt(result.shape.endIndex - 2, result.shape.endIndex - 1)
        if T.self == Float.self {
            var outputData = [T](repeating: 0, count: totalElements)
            for batch in 0..<batchCount {
                let offset = batch * rows * cols
                data.withUnsafeBufferPointer { sdata in
                    outputData.withUnsafeMutableBufferPointer { odata in
                        vDSP_mtrans(sdata.baseAddress! as! UnsafePointer<Float> + offset, 1, odata.baseAddress! as! UnsafeMutablePointer<Float> + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                    }
                }
            }
            result.data = outputData
        } else if T.self == Double.self {
            var outputData = [T](repeating: 0, count: totalElements)
            for batch in 0..<batchCount {
                let offset = batch * rows * cols
                data.withUnsafeBufferPointer { sdata in
                    outputData.withUnsafeMutableBufferPointer { odata in
                        vDSP_mtransD(sdata.baseAddress! as! UnsafePointer<Double> + offset, 1, odata.baseAddress! as! UnsafeMutablePointer<Double> + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                    }
                }
            }
            result.data = outputData
        } else {
            let compData = data.compactMap{ Float($0) }
            var outputData = [Float](repeating: 0, count: totalElements)
            for batch in 0..<batchCount {
                let offset = batch * rows * cols
                compData.withUnsafeBufferPointer { sdata in
                    outputData.withUnsafeMutableBufferPointer { odata in
                        vDSP_mtrans(sdata.baseAddress! + offset, 1, odata.baseAddress! + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                    }
                }
            }
            result.data = outputData.compactMap{ T($0) }
        }
        result.operation = "Transpose"
        result.parents = [
            (self, { v in
                print("Trans")
                print(v.gradient!)
                print(v.shape)
                return TensorKit.transpose(v.gradient!, shape: v.shape)})
        ]
        return result
    }
}
