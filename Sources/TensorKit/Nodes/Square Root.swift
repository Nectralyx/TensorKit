//
//  Square Root.swift
//  Synapse
//
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import Foundation
import Accelerate

@inlinable
public func sqrt<T: TensorType>(_ input: Tensor<T>) -> Tensor<T> {
    let result = Tensor<T>(.empty, shape: input.shape, calculate_grad: input.gradient != nil)
    var size = Int32(input.dataSize)
    if T.self == Float.self {
        var output = [Float](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvsqrtf(obuffer.baseAddress!, ibuffer.baseAddress! as! UnsafePointer<Float>, &size)
            }
        }
        result.data = output as! [T]
    } else if T.self == Double.self {
        var output = [Double](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvsqrt(obuffer.baseAddress!, ibuffer.baseAddress! as! UnsafePointer<Double>, &size)
            }
        }
        result.data = output as! [T]
    } else /*if T.self == Float16.self*/ {
        var output = [Float](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        let input = input.data.map{ Float($0) }
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.withUnsafeBufferPointer{ ibuffer in
                vvsqrtf(obuffer.baseAddress!, ibuffer.baseAddress!, &size)
            }
        }
        result.data = output.map{ T($0) }
    }
    result.parents = [
        (input, { v in
            (Tensor<T>(1) / (Tensor<T>(2) * v) * Tensor(v.gradient!, shape: v.shape)).data
        })
    ]
    result.operation = "Square Root"
    return result
}
