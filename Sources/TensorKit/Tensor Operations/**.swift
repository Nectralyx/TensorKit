//
//  **.swift
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
    static func **(lhs: Tensor, rhs: Tensor) -> Tensor {
        guard lhs.shape.last! == rhs.shape.dropLast().last! else {
            fatalError("Cannot perform matrix multiplication between tensors of shapes \(lhs.shape) and \(rhs.shape)")
       }
        while lhs.shape.count < rhs.shape.count {
            lhs.shape.insert(1, at: 0)
        }
        while rhs.shape.count < lhs.shape.count {
            rhs.shape.insert(1, at: 0)
        }
        let final: [Int] = zip(lhs.shape.dropLast(2), rhs.shape.dropLast(2)).map {
            if $0.0 == $0.1 {
                return $0.0
            } else if $0.0 > $0.1 && $0.1 == 1 {
               return $0.0
            } else if $0.1 > $0.0 && $0.0 == 1 {
               return $0.1
            } else {
                fatalError("Could not access for some reason: \(lhs.shape), \(rhs.shape)")
            }
        } + [lhs.shape[lhs.shape.count - 2], rhs.shape[rhs.shape.count - 1]]
        
        var finalShape = final.flatMap{ $0 }
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
                var outputData = [T](repeating: 0, count: result.dataSize)
                for batch in 0..<batchCount {
                    let lOffset = batch * aRows * aCols
                    let rOffset = batch * aCols * bCols
                    let oOffset = batch * aRows * bCols
                    lhs.data.withUnsafeBufferPointer { lBuffer in
                        rhs.data.withUnsafeBufferPointer { rBuffer in
                            outputData.withUnsafeMutableBufferPointer { oBuffer in
                                vDSP_mmul(
                                    lBuffer.baseAddress! as! UnsafePointer<Float> + lOffset, 1,
                                    rBuffer.baseAddress! as! UnsafePointer<Float> + rOffset, 1,
                                    oBuffer.baseAddress! as! UnsafeMutablePointer<Float> + oOffset, 1,
                                    vDSP_Length(aRows),
                                    vDSP_Length(bCols),
                                    vDSP_Length(aCols)
                                )
                            }
                        }
                    }
                }
                result.data = outputData
            } else if T.self == Double.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                for batch in 0..<batchCount {
                    let lOffset = batch * aRows * aCols
                    let rOffset = batch * aCols * bCols
                    let oOffset = batch * aRows * bCols
                    lhs.data.withUnsafeBufferPointer { lBuffer in
                        rhs.data.withUnsafeBufferPointer { rBuffer in
                            outputData.withUnsafeMutableBufferPointer { oBuffer in
                                vDSP_mmulD(
                                    lBuffer.baseAddress! as! UnsafePointer<Double> + lOffset, 1,
                                    rBuffer.baseAddress! as! UnsafePointer<Double> + rOffset, 1,
                                    oBuffer.baseAddress! as! UnsafeMutablePointer<Double> + oOffset, 1,
                                    vDSP_Length(aRows),
                                    vDSP_Length(bCols),
                                    vDSP_Length(aCols)
                                )
                            }
                        }
                    }
                }
                result.data = outputData
            } else {
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
                                vDSP_mmul(
                                    lBuffer.baseAddress! as! UnsafePointer<Float> + lOffset, 1,
                                    rBuffer.baseAddress! as! UnsafePointer<Float> + rOffset, 1,
                                    oBuffer.baseAddress! as! UnsafeMutablePointer<Float> + oOffset, 1,
                                    vDSP_Length(aRows),
                                    vDSP_Length(bCols),
                                    vDSP_Length(aCols)
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
