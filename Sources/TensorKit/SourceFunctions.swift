//
//  SourceFunctions.swift
//  TensorKit
//
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */
import Metal
import Foundation
import Accelerate

infix operator **: MultiplicationPrecedence

public extension Array {
    @inlinable
    func inserting(_ newElement: Element, at i: Int) -> Array {
        var copy = self
        copy.insert(newElement, at: i)
        return copy
    }
}

@inlinable
public func softmax<T: TensorType>(_ x: [T]) -> [T] {
    // Find the maximum value in the array
        let maxVal = x.max() ?? 0.0
        
        // Subtract the maximum value from each element to prevent overflow
        let exps = x.map { exp(Double($0) - Double(maxVal)) }
        
        // Calculate the sum of exponentials
        let expSum = exps.reduce(0.0, +)
        
        // Divide each exponential by the sum to get the softmax probabilities
        let softmaxValues = exps.map { $0 / expSum }
        
    return softmaxValues.map{ T($0) }
}

@inlinable
func softmaxJacobian<T: TensorType>(_ Y: [T]) -> [[T]] {
    let n = Y.count
    var softmax = Array(repeating: Array(repeating: 0 as T, count: n), count: n)
    
    // Compute the softmax function
    //let softmaxY = expY.map { $0 / sumExpY }
    let softmaxY = Y
    // Compute the derivative of softmax
    for i in 0..<n {
        for j in 0..<n {
            if i == j {
                softmax[i][j] = softmaxY[i] * (1 - softmaxY[i])
            } else {
                softmax[i][j] = -softmaxY[i] * softmaxY[j]
            }
        }
    }
    
    return softmax
}

@inlinable
public func ReLU<T: TensorType>(_ x: T) -> T {
    return max(0, x)
}

// Define the derivative of the ReLU function
@inlinable
func ReLUDerivative<T: TensorType>(_ x: T) -> T {
    return x > 0 ? 1.0 : 0.0
}

@inlinable
public func Sigmoid<T: TensorType>(_ x: T) -> T {
    return T(1.0) / (T(1.0) + T(exp(-Double(x))))
}

@inlinable
func SigmoidDerivative<T: TensorType>(_ x: T) -> T {
    let s = Sigmoid(x)
    return s * (1.0 - s)
}

@inlinable
public func Tanh<T: TensorType>(_ x: T) -> T {
    return T((exp(Double(x)) - exp(-Double(x))) / (exp(Double(x)) + exp(-Double(x))))
}

@inlinable
func TanhDerivative<T: TensorType>(_ x: T) -> T {
    let t = Tanh(x)
    return 1.0 - t * t
}

@inlinable
public func LeakyReLU<T: TensorType>(_ x: T, alpha: T = 0.01) -> T {
    return x > 0 ? 1.0 : alpha * x
}

@inlinable
func LeakyReLUDerivative<T: TensorType>(_ x: T, alpha: T = 0.01) -> T {
    return x > 0 ?  1.0 : alpha
}

@inlinable
public func add<T: TensorType>(_ x: [T], _ y: [T]) -> [T] {
    guard x.count == y.count else {
        fatalError("Mismatching inputs to multiply() function: \(x.count) & \(y.count)")
    }
    let result = x.count
    if T.self == Float.self {
        var outputData = [Float](repeating: 0, count: x.count)
        
        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vadd(
                            lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                            rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(result)
                        )
                }
            }
        }
        
        return outputData as! [T]
    } else if T.self == Double.self {
        var outputData = [Double](repeating: 0, count: result)
        
        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vaddD(
                            lBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                            rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(result)
                        )
                }
            }
        }
        
        return outputData as! [T]
    } else /*if T.self == Float16.self*/ {
        var outputData = [Float](repeating: 0, count: result)
        
        let lDataFloat = x.compactMap { Float($0) }
        let rDataFloat = y.compactMap { Float($0) }
        
        lDataFloat.withUnsafeBufferPointer { lBuffer in
            rDataFloat.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vadd(
                            lBuffer.baseAddress!, 1,
                            rBuffer.baseAddress!, 1,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(result)
                        )
                }
            }
        }
        return outputData as! [T]
    }

}

@inlinable
public func multiply<T: TensorType>(_ x: [T], _ y: [T]) -> [T] {
    guard x.count == y.count else {
        fatalError("Mismatching inputs to multiply() function: \(x.count) & \(y.count)")
    }
    let result = x.count
    if T.self == Float.self {
        var outputData = [Float](repeating: 0, count: x.count)
        
        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vmul(
                            lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                            rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(result)
                        )
                }
            }
        }
        
        return outputData as! [T]
    } else if T.self == Double.self {
        var outputData = [Double](repeating: 0, count: result)
        
        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vmulD(
                            lBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                            rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(result)
                        )
                }
            }
        }
        
        return outputData as! [T]
    } else /*if T.self == Float16.self*/ {
        var outputData = [Float](repeating: 0, count: result)
        
        let lDataFloat = x.compactMap { Float($0) }
        let rDataFloat = y.compactMap { Float($0) }
        
        lDataFloat.withUnsafeBufferPointer { lBuffer in
            rDataFloat.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vmul(
                            lBuffer.baseAddress!, 1,
                            rBuffer.baseAddress!, 1,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(result)
                        )
                }
            }
        }
        return outputData as! [T]
    }
}

public protocol TensorType: Codable, BinaryFloatingPoint {}
extension Float: TensorType {}
extension Double: TensorType {}
