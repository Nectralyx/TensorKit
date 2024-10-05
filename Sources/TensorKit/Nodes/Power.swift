//
//  Power.swift
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

@usableFromInline
internal func hiddenpow<T: TensorType>(_ input: Tensor<T>, _ exp: T) -> Tensor<T> {
    var size = Int32(input.dataSize)
    let exponential = [T](repeating: exp, count: input.dataSize)
    let result = Tensor<T>(.empty, shape: input.shape)
    if T.self == Float.self {
        var output = [Float](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvpowf(obuffer.baseAddress!, exponential as! [Float], ibuffer.baseAddress! as! UnsafePointer<Float>, &size)
            }
        }
        result.data = output as! [T]
    } else if T.self == Double.self {
        var output = [Double](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvpow(obuffer.baseAddress!, exponential as! [Double], ibuffer.baseAddress! as! UnsafePointer<Double>, &size)
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
                vvpowf(obuffer.baseAddress!, exponential.map{ Float($0) }, ibuffer.baseAddress!, &size)
            }
        }
        result.data = output.map{ T($0) }
    }
    return result
}

@inlinable
public func pow<T: TensorType>(_ input: Tensor<T>, _ exp: T) -> Tensor<T> {
    var size = Int32(input.dataSize)
    let exponential = [T](repeating: exp, count: input.dataSize)
    let result = Tensor<T>(.empty, shape: input.shape, calculate_grad: input.gradient != nil)
    if T.self == Float.self {
        var output = [Float](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvpowf(obuffer.baseAddress!, exponential as! [Float], ibuffer.baseAddress! as! UnsafePointer<Float>, &size)
            }
        }
        result.data = output as! [T]
    } else if T.self == Double.self {
        var output = [Double](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvpow(obuffer.baseAddress!, exponential as! [Double], ibuffer.baseAddress! as! UnsafePointer<Double>, &size)
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
                vvpowf(obuffer.baseAddress!, exponential.map{ Float($0) }, ibuffer.baseAddress!, &size)
            }
        }
        result.data = output.map{ T($0) }
    }
    result.operation = "Power (n^\(exp))"
    result.parents = [
        (input, { v in
            (hiddenpow(input, exp - 1) * Tensor(exp) * Tensor(v.gradient!, shape: v.shape)).data
        })
    ]
    return result
}

@usableFromInline
internal func hiddenpow<T: TensorType>(_ input: Tensor<T>, _ exp: Tensor<T>) -> Tensor<T> {
    let finalShape = mergeShapes(exp.shape, input.shape)
    let exp = exp.expand(to: finalShape)
    let input = input.expand(to: finalShape)
    var size = Int32(input.dataSize)
    let exponential = exp.data
    let result = Tensor<T>(.empty, shape: finalShape)
    if T.self == Float.self {
        var output = [Float](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvpowf(obuffer.baseAddress!, exponential as! [Float], ibuffer.baseAddress! as! UnsafePointer<Float>, &size)
            }
        }
        result.data = output as! [T]
    } else if T.self == Double.self {
        var output = [Double](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvpow(obuffer.baseAddress!, exponential as! [Double], ibuffer.baseAddress! as! UnsafePointer<Double>, &size)
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
                vvpowf(obuffer.baseAddress!, exponential.map{ Float($0) }, ibuffer.baseAddress!, &size)
            }
        }
        result.data = output.map{ T($0) }
    }
    return result
}

@inlinable
public func pow<T: TensorType>(_ input: Tensor<T>, _ exp: Tensor<T>) -> Tensor<T> {
    let finalShape = mergeShapes(exp.shape, input.shape)
    let exp = exp.expand(to: finalShape)
    let input = input.expand(to: finalShape)
    var size = Int32(input.dataSize)
    let exponential = exp.data
    let result = Tensor<T>(.empty, shape: finalShape, calculate_grad: input.gradient != nil)
    if T.self == Float.self {
        var output = [Float](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvpowf(obuffer.baseAddress!, exponential as! [Float], ibuffer.baseAddress! as! UnsafePointer<Float>, &size)
            }
        }
        result.data = output as! [T]
    } else if T.self == Double.self {
        var output = [Double](unsafeUninitializedCapacity: input.dataSize, initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        output.withUnsafeMutableBufferPointer{ obuffer in
            input.data.withUnsafeBufferPointer{ ibuffer in
                vvpow(obuffer.baseAddress!, exponential as! [Double], ibuffer.baseAddress! as! UnsafePointer<Double>, &size)
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
                vvpowf(obuffer.baseAddress!, exponential.map{ Float($0) }, ibuffer.baseAddress!, &size)
            }
        }
        result.data = output.map{ T($0) }
    }
    result.operation = "Power (n^\(exp))"
    result.parents = [
        (input, { v in
            (hiddenpow(input, exp - Tensor<T>(1)) * exp * Tensor(v.gradient!, shape: v.shape)).data
        })
    ]
    return result
}
