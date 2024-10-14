//
//  Cosine.swift
//  Nucleus
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import Foundation
import Accelerate

@inlinable
public func cos<T: TensorType>(_ x: Tensor<T>) -> Tensor<T> {
    var totalSize = Int32(x.dataSize)
    let result = Tensor<T>(.empty, shape: x.shape, calculate_grad: x.gradient != nil ? true : false)
    var outputData = [T](repeating: 0, count: x.dataSize)
    result.parents = [
        (x, { _ in multiply(sin(x.data), s: -1.0) })
    ]
    result.operation = "Cosine"
    if T.self == Float.self {
        x.data.withUnsafeBufferPointer { xBuffer in
            outputData.withUnsafeMutableBufferPointer { yBuffer in
                vvcosf(yBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                       xBuffer.baseAddress! as! UnsafePointer<Float>,
                       &totalSize
                )
            }
        }
        result.data = outputData
        return result
    } else if T.self == Double.self {
        x.data.withUnsafeBufferPointer { xBuffer in
            result.data.withUnsafeMutableBufferPointer { yBuffer in
                vvcos(yBuffer.baseAddress! as! UnsafeMutablePointer<Double>,
                       xBuffer.baseAddress! as! UnsafePointer<Double>,
                       &totalSize
                )
            }
        }
        result.data = outputData
        return result
    } else {
        x.data.withUnsafeBufferPointer { xBuffer in
            result.data.withUnsafeMutableBufferPointer { yBuffer in
                vvcosf(yBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                       xBuffer.baseAddress! as! UnsafePointer<Float>,
                       &totalSize
                )
            }
        }
        result.data = outputData
        return result
    }
}

