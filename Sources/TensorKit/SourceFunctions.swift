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

@usableFromInline
internal func repeatArray<T: TensorType>(_ array: [T], count: Int) -> [T] {
    // Calculate the total length of the resulting array
    let tt1 = CFAbsoluteTimeGetCurrent()
    let repeatedLength = array.count * count
    if T.self == Float.self {
        // Create an output array with the required length, initialized to zero
        let t1 = CFAbsoluteTimeGetCurrent()
        var result = [Float](repeating: 0, count: repeatedLength)
        let t2 = CFAbsoluteTimeGetCurrent()
        // Use unsafe mutable buffer pointers for efficient copying
        result.withUnsafeMutableBufferPointer { resultPointer in
            // Get a pointer to the start of the result buffer
            //let resultBase = resultPointer.baseAddress!
            guard let resultBase = resultPointer.baseAddress else {
                fatalError("Result base address is nil")
            }
            // Copy the original array into the result buffer for the first time
            array.withUnsafeBufferPointer { arrayPointer in
                // Get a pointer to the start of the array buffer
                guard let arrayBase = arrayPointer.baseAddress else {
                    fatalError("Array base address is nil")
                }
                let t3 = CFAbsoluteTimeGetCurrent()
                // Copy the original array into the result buffer
                vDSP_mmov(arrayBase as! UnsafePointer<Float>, resultBase, vDSP_Length(array.count), 1, vDSP_Length(array.count), 1)
                // Repeat the copy operation using exponential growth
                let t4 = CFAbsoluteTimeGetCurrent()
                var currentLength = array.count
                while currentLength < repeatedLength {
                    let t5 = CFAbsoluteTimeGetCurrent()
                    // Double the size of copied elements each time
                    let remainingLength = min(currentLength, repeatedLength - currentLength)
                    let t6 = CFAbsoluteTimeGetCurrent()
                    vDSP_mmov(resultBase, resultBase + currentLength, vDSP_Length(remainingLength), 1, vDSP_Length(remainingLength), 1)
                    let t7 = CFAbsoluteTimeGetCurrent()
                    currentLength += remainingLength
                    let t8 = CFAbsoluteTimeGetCurrent()
                    print("calculated reamining length: \(t6 - t5)")
                    print("processed copy: \(t7 - t6)")
                    print("calculated remaining length again: \(t8 - t7)")
                }
                print("repeatArray() DIAGNOSTIC")
                print("Created result array: \(t2 - t1)")
                print("made safety checks: \(t3 - t2)")
                print("processed first copy: \(t4 - t3)")
                print("processed more copies: \(CFAbsoluteTimeGetCurrent() - t4)")
            }
        }
        
        
        let tt2 = CFAbsoluteTimeGetCurrent()
        let out = result as! [T]
        print("Total repeatArray() time: \(tt2 - tt1)")
        return out
    } else if T.self == Double.self {
        // Create an output array with the required length, initialized to zero
        var result = [Double](repeating: 0, count: repeatedLength)
        
        // Use unsafe mutable buffer pointers for efficient copying
        result.withUnsafeMutableBufferPointer { resultPointer in
            // Get a pointer to the start of the result buffer
            let resultBase = resultPointer.baseAddress!
            
            // Copy the original array into the result buffer for the first time
            array.withUnsafeBufferPointer { arrayPointer in
                // Get a pointer to the start of the array buffer
                let arrayBase = arrayPointer.baseAddress!
                
                // Copy the original array into the result buffer
                vDSP_mmovD(arrayBase as! UnsafePointer<Double>, resultBase, vDSP_Length(array.count), 1, vDSP_Length(array.count), 1)
                
                // Repeat the copy operation using exponential growth
                var currentLength = array.count
                while currentLength < repeatedLength {
                    // Double the size of copied elements each time
                    let remainingLength = min(currentLength, repeatedLength - currentLength)
                    vDSP_mmovD(resultBase, resultBase + currentLength, vDSP_Length(remainingLength), 1, vDSP_Length(remainingLength), 1)
                    currentLength += remainingLength
                }
            }
        }
        
        return result as! [T]
    } else {
        // Create an output array with the required length, initialized to zero
        var result = [Float](repeating: 0, count: repeatedLength)
        
        // Use unsafe mutable buffer pointers for efficient copying
        result.withUnsafeMutableBufferPointer { resultPointer in
            // Get a pointer to the start of the result buffer
            let resultBase = resultPointer.baseAddress!
            
            // Copy the original array into the result buffer for the first time
            array.map{ Float($0) }.withUnsafeBufferPointer { arrayPointer in
                // Get a pointer to the start of the array buffer
                let arrayBase = arrayPointer.baseAddress!
                
                // Copy the original array into the result buffer
                vDSP_mmov(arrayBase, resultBase, vDSP_Length(array.count), 1, vDSP_Length(array.count), 1)
                
                // Repeat the copy operation using exponential growth
                var currentLength = array.count
                while currentLength < repeatedLength {
                    // Double the size of copied elements each time
                    let remainingLength = min(currentLength, repeatedLength - currentLength)
                    vDSP_mmov(resultBase, resultBase + currentLength, vDSP_Length(remainingLength), 1, vDSP_Length(remainingLength), 1)
                    currentLength += remainingLength
                }
            }
        }
        
        return result.map { T($0) }
    }
}

@inlinable
public func canExpand(_ a: [Int], _ b: [Int]) -> Bool {
     for (x, y) in zip(a, b) {
         if x == y || x == 1 || y == 1 {
             continue
         } else {
             return false
         }
     }
     return true
 }

@inlinable
public func appendVector<T: FloatingPoint>(_ source: [T], to destination: inout [T]) {
    let sourceCount = source.count
    let destinationCount = destination.count
    
    // Pre-allocate memory for the destination vector to fit both the old and new elements
    destination.reserveCapacity(destinationCount + sourceCount)
    
    // Extend destination with zeroes temporarily to ensure there's enough space
    destination.append(contentsOf: repeatElement(0 as T, count: sourceCount))
    
    // Use Accelerate's vDSP_mmov to copy the source data into the end of the destination vector
    source.withUnsafeBufferPointer { srcBuffer in
        destination.withUnsafeMutableBufferPointer { destBuffer in
            vDSP_mmov(srcBuffer.baseAddress! as! UnsafePointer<Float>, destBuffer.baseAddress! as! UnsafeMutablePointer<Float> + destinationCount, vDSP_Length(sourceCount), 1, vDSP_Length(sourceCount), 1)
        }
    }
}
