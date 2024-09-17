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
import cxxLibrary

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


public protocol TensorType: Codable, BinaryFloatingPoint {}
extension Float: TensorType {}
extension Double: TensorType {}

/*@usableFromInline
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
}*/

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
    
    // Allocate a new array with enough capacity to hold both the original and new data
    var result = [T](repeating: 0, count: destinationCount + sourceCount)
    
    // Use vDSP_mmov to copy the original destination data into the result array
    destination.withUnsafeBufferPointer { destBuffer in
        result.withUnsafeMutableBufferPointer { resultBuffer in
            vDSP_mmov(destBuffer.baseAddress! as! UnsafePointer<Float>, resultBuffer.baseAddress! as! UnsafeMutablePointer<Float>, vDSP_Length(destinationCount), 1, vDSP_Length(destinationCount), 1)
        }
    }
    
    // Use vDSP_mmov to copy the source data into the result array after the original data
    source.withUnsafeBufferPointer { srcBuffer in
        result.withUnsafeMutableBufferPointer { resultBuffer in
            vDSP_mmov(srcBuffer.baseAddress! as! UnsafePointer<Float>, resultBuffer.baseAddress! as! UnsafeMutablePointer<Float> + destinationCount, vDSP_Length(sourceCount), 1, vDSP_Length(sourceCount), 1)
        }
    }
    
    // Replace the destination array with the result
    destination = result
}



//testing(Int32(a), Int32(result))
