//
//  Softmax.swift
//  TensorKit
//
//
/*
* Copyright (c) 2024 Nectralyx.
* This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*/

import Foundation
import TKCore
import Accelerate

@usableFromInline
internal func softmaxJacobian<T: TensorType>(_ y: Tensor<T>, outputGrad: [T], _ dimension: Int) -> [T] {
    let t1o = CFAbsoluteTimeGetCurrent()
    let jSize = y.shape[dimension]
    var outputs = [T](repeating: 0, count: outputGrad.count)
    let blockStride = y.shape.dropFirst(dimension + 1).reduce(1, *)
    if T.self == Float.self {
        var t1 = CFAbsoluteTimeGetCurrent()
        var t2 = CFAbsoluteTimeGetCurrent()
        y.data.withUnsafeBufferPointer{ xBuffer in
            outputGrad.withUnsafeBufferPointer{ oBuffer in
                outputs.withUnsafeMutableBufferPointer{ yBuffer in
                    y.shape.withUnsafeBufferPointer{ sBuffer in
                        t1 = CFAbsoluteTimeGetCurrent()
                        softmaxJacobian(
                            xBuffer.baseAddress! as? UnsafePointer<Float>,
                            yBuffer.baseAddress! as? UnsafeMutablePointer<Float>,
                            oBuffer.baseAddress! as? UnsafePointer<Float>,
                            Int32(dimension),
                            sBuffer.baseAddress! as? UnsafePointer<Int32>,
                            Int32(jSize),
                            Int32(y.dataSize),
                            Int32(blockStride)
                        )
                        t2 = CFAbsoluteTimeGetCurrent()
                    }
                }
            }
        }
        let t2o = CFAbsoluteTimeGetCurrent()
        print("Softmax C++ function took \(t2 - t1) seconds of the total function time: \(t2o - t1o) seconds")
    } else if T.self == Double.self {
        y.data.withUnsafeBufferPointer{ xBuffer in
            outputGrad.withUnsafeBufferPointer{ oBuffer in
                outputs.withUnsafeMutableBufferPointer{ yBuffer in
                    y.shape.withUnsafeBufferPointer{ sBuffer in
                        softmaxJacobianD(
                            xBuffer.baseAddress! as? UnsafePointer<Double>,
                            yBuffer.baseAddress! as? UnsafeMutablePointer<Double>,
                            oBuffer.baseAddress! as? UnsafePointer<Double>,
                            Int32(dimension),
                            sBuffer.baseAddress! as? UnsafePointer<Int32>,
                            Int32(jSize),
                            Int32(y.dataSize),
                            Int32(blockStride)
                        )
                    }
                }
            }
        }
    }
    return outputs
}

@inlinable
public func Softmax<T: TensorType>(_ input: Tensor<T>, dimension: Int) -> Tensor<T> {
    // Ensure the dimension is valid
    precondition(dimension < input.shape.count, "Invalid dimension for softmax")
    
    let shape = input.shape
    let dimSize = shape[dimension] // This will be the length of each softmax vector
    var softmaxValues = [T](repeating: 0.0, count: input.data.count)
    
    // Calculate the stride and number of blocks for the specified dimension
    let numBlocks = input.dataSize / dimSize // This is the number of times that softmax will be applied
    let blockStride = shape.dropFirst(dimension + 1).reduce(1, *)
    
    if T.self == Float.self {
        softmaxValues.withUnsafeMutableBufferPointer{ oBuffer in
            input.data.withUnsafeBufferPointer{ iBuffer in
                input.shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                    TKCore.softmax(
                        iBuffer.baseAddress! as? UnsafePointer<Float>,
                        oBuffer.baseAddress! as? UnsafeMutablePointer<Float>,
                        sBuffer.baseAddress!,
                        Int32(dimension),
                        Int32(dimSize),
                        Int32(numBlocks),
                        Int32(blockStride)
                    )
                }
            }
        }
    } else if T.self == Double.self {
        softmaxValues.withUnsafeMutableBufferPointer{ oBuffer in
            input.data.withUnsafeBufferPointer{ iBuffer in
                input.shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                    TKCore.softmaxD(
                        iBuffer.baseAddress! as? UnsafePointer<Double>,
                        oBuffer.baseAddress! as? UnsafeMutablePointer<Double>,
                        sBuffer.baseAddress!,
                        Int32(dimension),
                        Int32(dimSize),
                        Int32(numBlocks),
                        Int32(blockStride)
                    )
                }
            }
        }
    }
    
    let result = Tensor(softmaxValues, shape: input.shape, calculate_grad: input.gradient != nil)
    result.operation = "Softmax"
    result.parents = [
        (input, { v in
            softmaxJacobian(result, outputGrad: v.gradient!, dimension)
        })
    ]
    return result
}
