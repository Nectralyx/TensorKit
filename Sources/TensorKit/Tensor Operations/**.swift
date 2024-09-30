//
//  **.swift
//  Synapse
//
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import Foundation
import TKCore

public extension Tensor {
    @inlinable
    static func **(lhs: Tensor, rhs: Tensor) -> Tensor {
        guard lhs.shape.last! == rhs.shape.dropLast().last! else {
            return Tensor(shape: [])
       }
        
        var finalShape = mergeShapes(lhs.shape, rhs.shape)
        var lefttargetSize = finalShape
        lefttargetSize.removeLast(2)
        lefttargetSize.append(contentsOf: [lhs.shape.dropLast().last!, lhs.shape.last!])
        var righttargetSize = finalShape
        righttargetSize.removeLast(2)
        righttargetSize.append(contentsOf: [rhs.shape.dropLast().last!, rhs.shape.last!])
        let lhs = lhs.expand(to: lefttargetSize)
        let rhs = rhs.expand(to: righttargetSize)
        finalShape[finalShape.endIndex - 2] = lhs.shape.dropLast().last!
        finalShape[finalShape.endIndex - 1] = rhs.shape.last!
        let result = Tensor(.empty, shape: finalShape, calculate_grad: (lhs.gradient != nil || rhs.gradient != nil) ? true : false)
        let aRows = lhs.shape.dropLast().last!
        let aCols = lhs.shape.last!
        let bCols = rhs.shape.last!
        let batchCount = lhs.shape.dropLast(2).reduce(1, *)
        
            if T.self == Float.self {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                for batch in 0..<batchCount {
                    let lOffset = batch * aRows * aCols
                    let rOffset = batch * aCols * bCols
                    let oOffset = batch * aRows * bCols
                    lhs.data.withUnsafeBufferPointer { lBuffer in
                        rhs.data.withUnsafeBufferPointer { rBuffer in
                            outputData.withUnsafeMutableBufferPointer { oBuffer in
                                matrixmultiply(
                                    lBuffer.baseAddress! as! UnsafePointer<Float> + lOffset,
                                    rBuffer.baseAddress! as! UnsafePointer<Float> + rOffset,
                                    Int32(aRows),
                                    Int32(aCols),
                                    Int32(bCols),
                                    oBuffer.baseAddress! + oOffset
                                )
                            }
                        }
                    }
                }
                result.data = outputData as! [T]
            } else if T.self == Double.self {
                var outputData = [Double](repeating: 0, count: result.dataSize)
                for batch in 0..<batchCount {
                    let lOffset = batch * aRows * aCols
                    let rOffset = batch * aCols * bCols
                    let oOffset = batch * aRows * bCols
                    lhs.data.withUnsafeBufferPointer { lBuffer in
                        rhs.data.withUnsafeBufferPointer { rBuffer in
                            outputData.withUnsafeMutableBufferPointer { oBuffer in
                                matrixmultiplyD(
                                    lBuffer.baseAddress! as! UnsafePointer<Double> + lOffset,
                                    rBuffer.baseAddress! as! UnsafePointer<Double> + rOffset,
                                    Int32(aRows),
                                    Int32(aCols),
                                    Int32(bCols),
                                    oBuffer.baseAddress! + oOffset
                                )
                            }
                        }
                    }
                }
                result.data = outputData as! [T]
            } else /*if T.self == Float16.self*/ {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                let lDataFloat = lhs.data.compactMap { Float($0) }
                let rDataFloat = rhs.data.compactMap { Float($0) }
                for batch in 0..<batchCount {
                    let lOffset = batch * aRows * aCols
                    let rOffset = batch * aCols * bCols
                    let oOffset = batch * aRows * bCols
                    lDataFloat.withUnsafeBufferPointer { lBuffer in
                        rDataFloat.withUnsafeBufferPointer { rBuffer in
                            outputData.withUnsafeMutableBufferPointer { oBuffer in
                                matrixmultiply(
                                    lBuffer.baseAddress! + lOffset,
                                    rBuffer.baseAddress! + rOffset,
                                    Int32(aRows),
                                    Int32(aCols),
                                    Int32(bCols),
                                    oBuffer.baseAddress! + oOffset
                                )
                            }
                        }
                    }
                }
            }
        
        func swapIndices(_ shape: [Int]) -> [Int] {
            var new = shape
            new[new.endIndex - 1] = shape.dropLast().last!
            new[new.endIndex - 2] = shape.last!
            return new
        }
        
        result.operation = "**"
        result.parents = [
            (lhs, { v in
                return matrixMultiply(v.gradient!, TensorKit.transpose(rhs.data, shape: rhs.shape), aShape: v.shape, bShape: swapIndices(rhs.shape))
            }),
            (rhs, { v in
                return matrixMultiply(TensorKit.transpose(lhs.data, shape: lhs.shape), v.gradient!, aShape: swapIndices(lhs.shape), bShape: v.shape)
            })
        ]
        
        return result
    }
}
