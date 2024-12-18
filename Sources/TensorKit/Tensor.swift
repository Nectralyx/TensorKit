//
//  Tensor.swift
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
import TKCore

public enum TensorInitialization: Codable {
    case ones
    case zeros
    case mean_scaling
    case xavier_glorot
    case he
    case random
    case random_small
    case empty
}

open class Tensor<T: TensorComplex>: Codable, CustomStringConvertible, Sequence, Hashable, Equatable {
    public static func ==(lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.shape == rhs.shape && lhs.data == rhs.data && lhs.gradient == rhs.gradient
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(shape)
        hasher.combine(data)
        hasher.combine(gradient)
        hasher.combine(operation)
    }
    public var description: String {
        return formatTensor(data, shape: shape)
    }
    public var shape: [Int] // Array to represent the shape of the tensor
    public var data: [T]
    public var operation: String? = nil
    public var parents: [(Tensor, (Tensor) -> [T])] = []
    public var gradient: [T]?
    //var calculate_grad: Bool = false
    public var dataSize: Int {
        return shape.reduce(1, *)
    }
    @inlinable
    public init(_ input: Tensor) {
        self.shape = input.shape
        self.data = input.data
        self.gradient = input.gradient
        self.operation = "Clone Of: \(input.operation)"
        self.parents = input.parents
    }
    @inlinable
    public init(_ input: [[[T]]], calculate_grad: Bool = false) {
        self.shape = [input.count, input[0].count, input[0][0].count]
        self.data = input.flatMap { $0.flatMap { $0 } }
        self.gradient = calculate_grad ? [T](repeating: 0.0, count: data.count) : nil
        //self.calculate_grad = calculate_grad
    }
    @inlinable
    public init(_ input: [[T]], calculate_grad: Bool = false) {
        self.shape = [input.count, input[0].count]
        self.data = input.flatMap{ $0 }
        self.gradient = calculate_grad ? [T](repeating: 0.0, count: data.count) : nil
        //self.calculate_grad = calculate_grad
    }
    @inlinable
    public init(_ input: [T], shape: [Int], calculate_grad: Bool = false) {
        self.shape = shape
        self.data = input
        self.gradient = calculate_grad ? [T](repeating: 0.0, count: data.count) : nil
        //self.calculate_grad = calculate_grad
    }
    @inlinable
    public init(_ input: T, calculate_grad: Bool = false) {
        self.shape = [1]
        self.data = [input]
        self.gradient = calculate_grad ? [T](repeating: 0.0, count: data.count) : nil
        //self.calculate_grad = calculate_grad
    }
    @inlinable
    public init(_ initializer: TensorInitialization = .zeros, shape: [Int], calculate_grad: Bool = false) {
        self.shape = shape
        switch initializer {
        case .zeros:
            self.data = [T](repeating: 0, count: shape.reduce(1, *))
        case .ones:
            self.data = [T](repeating: 1, count: shape.reduce(1, *))
        case .mean_scaling:
            let scale = T(1 / shape.reduce(1, *))
            self.data = [T](repeating: scale, count: shape.reduce(1, *))
        case .xavier_glorot:
            let xavierScale = shape.count >= 2 ? sqrt(6.0 / Double(shape[shape.endIndex - 1] + shape[shape.endIndex - 2])) : sqrt(6.0 / Double(shape[shape.endIndex - 1] + 1))
            let range = -xavierScale...xavierScale
            self.data = [T](repeating: T(Double.random(in: -xavierScale...xavierScale)), count: shape.reduce(1, *))
            //self.data = [T](repeating: 1, count: shape.reduce(1, *))
            self.data.map {$0 + T(Double.random(in: range))}
        case .he:
            let heScale = sqrt(6.0 / Double(shape[shape.endIndex - 2]))
            self.data = [T](repeating: T(Double.random(in: -heScale...heScale)), count: shape.reduce(1, *))
        case .random:
            self.data = []
            for _ in 0..<shape.reduce(1, *) {
                self.data.append(T(Double.random(in: -10000...10000)))
            }
            //self.data = [T](repeating: T(Double.random(in: -10000...10000)), count: shape.reduce(1, *))
        case .empty:
            self.data = []
        case .random_small:
            self.data = []
            for _ in 0..<shape.reduce(1, *) {
                self.data.append(T(Double.random(in: -0.01...0.01)))
            }
        }
        self.gradient = calculate_grad ? [T](repeating: 0.0, count: shape.reduce(1, *)) : nil
       // self.calculate_grad = calculate_grad
    }
    open func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(data, forKey: .data)
        try container.encode(shape, forKey: .shape)
        try container.encode(operation, forKey: .operation)
        try container.encode(gradient, forKey: .gradient)
        //try container.encode(calculate_grad, forKey: .calculate_grad)
    }
    public required init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.data = try container.decode([T].self, forKey: .data)
        self.shape = try container.decode([Int].self, forKey: .shape)
        self.operation = try container.decode(String?.self, forKey: .operation)
        self.gradient = try container.decode([T]?.self, forKey: .gradient)
        //self.calculate_grad = try container.decode(Bool.self, forKey: .calculate_grad)
    }
    enum CodingKeys: CodingKey {
        case data
        case shape
        case operation
        case gradient
       // case calculate_grad
    }
    // Recursive function to format the tensor
    private func formatTensor(_ data: [T], shape: [Int], depth: Int = 0) -> String {
        // Base case: if shape has only one dimension, print the array directly
            
        if dataSize == 1 {
            return "\(String(Float(data[0])))"
        }
        if shape.count == 1 {
            return "[\(data.map { String(Float($0)) }.joined(separator: ", "))]"
        }

        // Recursive case: format each sub-array based on the shape
        let size = shape[0]
        let subShape = Array(shape.dropFirst())
        let stride = subShape.reduce(1, *) // Number of elements in each sub-array
        var result = "[\n"  // Opening bracket on a new line
        let indent = String(repeating: "  ", count: depth + 1) // Indentation for elements
        
        for i in 0..<size {
            let start = i * stride
            let end = (i + 1) * stride
            let subArray = Array(data[start..<end])
                
            // Recursively format each sub-array
            result += indent + formatTensor(subArray, shape: subShape, depth: depth + 1)

            // Add comma and newline for all but the last element
            if i < size - 1 {
                result += ",\n"
            }
        }

        result += "\n" + String(repeating: "  ", count: depth) + "]"  // Closing bracket on a new line
        return result
    }
    
    public func makeIterator() -> IndexingIterator<[T]> {
        return data.makeIterator()
    }

    // Sum over a specified dimension
    @inlinable
    public func sum(along dimension: Int, keepDims: Bool = true) -> Tensor {
        guard shape[dimension] > 0 else { return self }
        var newDimensions = shape
        if keepDims {
            newDimensions[dimension] = 1
        } else {
            newDimensions.remove(at: dimension)
        }
        // Ensure the dimension is valid
        precondition(dimension < shape.count, "Invalid dimension for sum")

        let dimSize = shape[dimension]
        var summedValues = [T](repeating: 0.0, count: data.count / dimSize)

        // Calculate the stride and number of blocks for the specified dimension
        let numBlocks = data.count / dimSize
        let blockStride = shape.dropFirst(dimension + 1).reduce(1, *)

        // Perform the summation along the specified dimension
        for blockIndex in 0..<numBlocks {
            // Compute the starting index for this block
            let blockStartIndex = (blockIndex / blockStride) * (blockStride * dimSize) + (blockIndex % blockStride)

            for j in 0..<dimSize {
                let index = blockStartIndex + j * blockStride
                summedValues[blockIndex] += data[index]
            }
        }
        let result = Tensor(summedValues, shape: newDimensions, calculate_grad: gradient != nil)
        result.operation = "Sum"
        result.parents = [
            (self, { v in
                return Tensor(v.gradient!, shape: v.shape).expand(to: self.shape).data
            })
        ]
        return result
    }
    
    @inlinable
    public func sum(along: [Int], keepDims: Bool = true) -> Tensor {
        var result = self
        for i in along {
            result = result.sum(along: i, keepDims: keepDims)
        }
        return result
    }

    @inlinable
    public func sum(keepDims: Bool = true) -> Tensor {
        var result = Tensor(.empty, shape: keepDims ? [Int](repeating: 1, count: shape.count) : [1], calculate_grad: gradient != nil)
        if T.self == Float.self {
            var output: Float = 0
            data.withUnsafeBufferPointer { ibuffer in
                vDSP_sve(ibuffer.baseAddress! as! UnsafePointer<Float>, 1, &output, vDSP_Length(dataSize))
            }
            result.data = [output] as! [T]
            result.operation = "Sum"
            result.parents = [
                (self, { v in Tensor(v.gradient!, shape: v.shape).expand(to: self.shape).data })
            ]
        } else if T.self == Double.self {
            var output: Double = 0
            data.withUnsafeBufferPointer { ibuffer in
                vDSP_sveD(ibuffer.baseAddress! as! UnsafePointer<Double>, 1, &output, vDSP_Length(dataSize))
            }
            result.data = [output] as! [T]
            result.operation = "Sum"
            result.parents = [
                (self, { v in Tensor(v.gradient!, shape: result.shape).expand(to: self.shape).data })
            ]
        }
        return result
    }
    
    // Reshape to new dimensions
    @inlinable
    public func reshape(to newDimensions: [Int]) -> Tensor {
        let totalElements = shape.reduce(1, *)
        let newTotalElements = newDimensions.reduce(1, *)
        if totalElements == newTotalElements {
            return Tensor(data, shape: newDimensions)
        }
        
        if totalElements < newTotalElements {
            return expand(to: newDimensions)
        } else if totalElements > newTotalElements {
            return Tensor(TensorKit.sum(data, shape: shape, to: newDimensions), shape: newDimensions)
        }
        fatalError("Could Not reshape data properly")
    }
    
    @inlinable
    public func view(_ shape: [Int]) -> Tensor {
        guard shape.reduce(1, *) == dataSize else {
            fatalError("Could not match data from \(self.shape) to \(shape). Did you mean to use Reshape?")
        }
        let result = Tensor(data, shape: shape, calculate_grad: self.gradient != nil)
        result.parents = [
            (self, { v in v.gradient! })
        ]
        return result
    }
    
    // Broadcast to new dimensions
    @inlinable
    public func expand(to targetDimensions: [Int]) -> Tensor {
        guard targetDimensions != shape else {
            return self
        }
        
        var newDimensions = shape
        var mainCandidate = newDimensions
        while mainCandidate.count < targetDimensions.count {
            mainCandidate.insert(1, at: 0)
        }
        
        if canExpand(mainCandidate, targetDimensions) {
            newDimensions = mainCandidate
        } else {
            while newDimensions.count < targetDimensions.count {
                newDimensions.insert(1, at: newDimensions.count)
            }
        }
        
        let gradientMap = newDimensions.enumerated().filter{ $0.element == 1 }.map{ $0.offset }
        var broadcastedData = data
        let dimCount = newDimensions.count
        for i in (0..<dimCount).reversed() {
            if newDimensions[i] == 1 && targetDimensions[i] > 1 {
                let intermediateData = broadcastedData
                var intermediateShape = newDimensions
                intermediateShape[i] = targetDimensions[i]
                var outputData = [T](repeating: 0, count: intermediateShape.reduce(1, *))
                if T.self == Float.self {
                    intermediateData.withUnsafeBufferPointer{ iBuffer in
                        outputData.withUnsafeMutableBufferPointer{ oBuffer in
                            expand_along_dimension(
                                iBuffer.baseAddress! as? UnsafePointer<Float>,
                                oBuffer.baseAddress! as? UnsafeMutablePointer<Float>,
                                newDimensions.map{ Int32($0) },
                                intermediateShape.map{ Int32($0) },
                                Int32(newDimensions.count),
                                Int32(i)
                            )
                        }
                    }
                } else if T.self == Double.self {
                    intermediateData.withUnsafeBufferPointer{ iBuffer in
                        outputData.withUnsafeMutableBufferPointer{ oBuffer in
                            expand_along_dimensionD(
                                iBuffer.baseAddress! as? UnsafePointer<Double>,
                                oBuffer.baseAddress! as? UnsafeMutablePointer<Double>,
                                newDimensions.map{ Int32($0) },
                                intermediateShape.map{ Int32($0) },
                                Int32(newDimensions.count),
                                Int32(i)
                            )
                        }
                    }
                }
                newDimensions[i] = targetDimensions[i]
                broadcastedData = outputData
            }
        }
        let result = Tensor(broadcastedData, shape: targetDimensions, calculate_grad: gradient != nil)
        broadcastedData
        result.operation = "Expand"
        result.parents = [
            (self, { v in TensorKit.sum(v.gradient!, shape: v.shape, along: gradientMap)})
        ]
        return result
    }
    
    @inlinable
    public func ravelIndex(_ indices: [Int]) -> Int {
        // Convert multi-dimensional indices to a flat index
        var flatIndex = 0
        var multiplier = 1
        for (i, index) in indices.reversed().enumerated() {
            flatIndex += index * multiplier
            multiplier *= shape[shape.count - 1 - i]
        }
        return flatIndex
    }
    
    @inlinable
    public func mean(_ dimension: Int, keepDims: Bool = true) -> Tensor {
        let rhs = Tensor(T(shape[dimension]))
        let output = self.sum(along: dimension, keepDims: keepDims) / rhs
        output.operation = "Mean"
        return output
    }
    
    @inlinable
    public func mean(_ dimensions: [Int], keepDims: Bool = true) -> Tensor {
        var total = 1
        for i in dimensions {
            total *= shape[i]
        }
        let rhs = Tensor(T(total))
        let output = self.sum(along: dimensions, keepDims: keepDims) / rhs
        output.operation = "Mean"
        return output
    }
    
    @inlinable
    public func mean(keepDims: Bool = true) -> Tensor {
        let rhs = Tensor(T(dataSize))
        let output = self.sum(keepDims: keepDims) / rhs
        output.operation = "Mean"
        return output
    }
    
    @inlinable
    public func variance(_ dimensions: [Int]) -> Tensor {
        var total = 1
        for i in dimensions {
            total *= shape[i]
        }
        let N = Tensor(1.0 / T(total))
        let subs = self - mean(dimensions)
        let src = pow(subs, 2)
        let srcShape = src.shape
        let sum = src.sum(along: dimensions) * N
        sum.parents = [
            (self, { v in
                let sumGrad = Tensor(multiply(v.gradient!, s: N.data[0]), shape: v.shape)
                let powGrad = sumGrad.expand(to: srcShape)
                return (powGrad * subs * Tensor(2)).data
            })
        ]
        sum.operation = "Variance"
        return sum
    }
    
    @inlinable
    public func permute(_ order: [Int]) -> Tensor {
        guard order.count == self.shape.count else {
            fatalError("Shape mismatch")
        }
        var result = [T](repeating: 0, count: dataSize)
        var gresult = [T](repeating: 0, count: dataSize)
        let originalStrides = generateStrides(shape)
        let newShape = order.map{ shape[$0] }
        let newStrides = generateStrides(newShape)
        if gradient != nil {
            if T.self == Float.self  {
                result.withUnsafeMutableBufferPointer{ result in
                    gresult.withUnsafeMutableBufferPointer{ gResult in
                        originalStrides.map{ Int32($0) }.withUnsafeBufferPointer{ osBuffer in
                            newStrides.map{ Int32($0) }.withUnsafeBufferPointer{ nsBuffer in
                                data.withUnsafeBufferPointer{ xBuffer in
                                    gradient!.withUnsafeBufferPointer{ qBuffer in
                                        shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                                            order.map{ Int32($0) }.withUnsafeBufferPointer{ oBuffer in
                                                TKCore.permute(
                                                    xBuffer.baseAddress! as? UnsafePointer<Float>,
                                                    qBuffer.baseAddress! as? UnsafePointer<Float>,
                                                    result.baseAddress! as? UnsafeMutablePointer<Float>,
                                                    gResult.baseAddress! as? UnsafeMutablePointer<Float>,
                                                    sBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                    oBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                    osBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                    nsBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                    Int32(dataSize),
                                                    Int32(shape.count)
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else if T.self  == Double.self {
                result.withUnsafeMutableBufferPointer{ result in
                    gresult.withUnsafeMutableBufferPointer{ gResult in
                        originalStrides.map{ Int32($0) }.withUnsafeBufferPointer{ osBuffer in
                            newStrides.map{ Int32($0) }.withUnsafeBufferPointer{ nsBuffer in
                                data.withUnsafeBufferPointer{ xBuffer in
                                    gradient!.withUnsafeBufferPointer{ qBuffer in
                                        shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                                            order.map{ Int32($0) }.withUnsafeBufferPointer{ oBuffer in
                                                TKCore.permuteD(
                                                    xBuffer.baseAddress! as? UnsafePointer<Double>,
                                                    qBuffer.baseAddress! as? UnsafePointer<Double>,
                                                    result.baseAddress! as? UnsafeMutablePointer<Double>,
                                                    gResult.baseAddress! as? UnsafeMutablePointer<Double>,
                                                    sBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                    oBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                    osBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                    nsBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                    Int32(dataSize),
                                                    Int32(shape.count)
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if T.self == Float.self  {
                result.withUnsafeMutableBufferPointer{ result in
                    originalStrides.map{ Int32($0) }.withUnsafeBufferPointer{ osBuffer in
                        newStrides.map{ Int32($0) }.withUnsafeBufferPointer{ nsBuffer in
                            data.withUnsafeBufferPointer{ xBuffer in
                                shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                                    order.map{ Int32($0) }.withUnsafeBufferPointer{ oBuffer in
                                        TKCore.permuteNoGrad(
                                            xBuffer.baseAddress! as? UnsafePointer<Float>,
                                            result.baseAddress! as? UnsafeMutablePointer<Float>,
                                            sBuffer.baseAddress! as? UnsafePointer<Int32>,
                                            oBuffer.baseAddress! as? UnsafePointer<Int32>,
                                            osBuffer.baseAddress! as? UnsafePointer<Int32>,
                                            nsBuffer.baseAddress! as? UnsafePointer<Int32>,
                                            Int32(dataSize),
                                            Int32(shape.count)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            } else if T.self == Double.self {
                result.withUnsafeMutableBufferPointer{ result in
                    originalStrides.map{ Int32($0) }.withUnsafeBufferPointer{ osBuffer in
                        newStrides.map{ Int32($0) }.withUnsafeBufferPointer{ nsBuffer in
                            data.withUnsafeBufferPointer{ xBuffer in
                                shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                                    order.map{ Int32($0) }.withUnsafeBufferPointer{ oBuffer in
                                        TKCore.permuteNoGradD(
                                            xBuffer.baseAddress! as? UnsafePointer<Double>,
                                            result.baseAddress! as? UnsafeMutablePointer<Double>,
                                            sBuffer.baseAddress! as? UnsafePointer<Int32>,
                                            oBuffer.baseAddress! as? UnsafePointer<Int32>,
                                            osBuffer.baseAddress! as? UnsafePointer<Int32>,
                                            nsBuffer.baseAddress! as? UnsafePointer<Int32>,
                                            Int32(dataSize),
                                            Int32(shape.count)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let output = Tensor(result, shape: newShape, calculate_grad: self.gradient != nil)
        output.gradient = gresult
        output.parents = [
            (self, { v in
                var inverseOrder = order
                for i in 0..<order.count {
                    inverseOrder[order[i]] = i
                }
                
               /* var adjustedShape = v.shape
                while adjustedShape.first == 1 && adjustedShape.count > order.count {
                    adjustedShape.removeFirst()
                }
                if adjustedShape.count != order.count {
                    fatalError("Order Does Not Match Shape. This error typically occurs from the transpose() function.")
                }*/
                
                let gnewShape = order.map{ v.shape[$0] }
                var result = [T](repeating: 0, count: v.dataSize)
                let originalStrides = generateStrides(v.shape)
                //print("NewShape: \(gnewShape) from: \(adjustedShape)")
                let newStrides = generateStrides(gnewShape)
                //print("new Strides: \(newStrides)")
                if T.self == Float.self  {
                    result.withUnsafeMutableBufferPointer{ result in
                        originalStrides.map{ Int32($0) }.withUnsafeBufferPointer{ osBuffer in
                            newStrides.map{ Int32($0) }.withUnsafeBufferPointer{ nsBuffer in
                                v.gradient!.withUnsafeBufferPointer{ xBuffer in
                                    output.shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                                        inverseOrder.map{ Int32($0) }.withUnsafeBufferPointer{ oBuffer in
                                            /*print("VALUES")
                                            print(result)
                                            print(originalStrides)
                                            print(newStrides)
                                            print(Tensor(v.gradient!, shape: v.shape))
                                            print(output.shape)
                                            print(inverseOrder)*/
                                            TKCore.permuteNoGrad(
                                                xBuffer.baseAddress! as? UnsafePointer<Float>,
                                                result.baseAddress! as? UnsafeMutablePointer<Float>,
                                                sBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                oBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                osBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                nsBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                Int32(v.dataSize),
                                                Int32(v.shape.count)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if T.self  == Double.self {
                    result.withUnsafeMutableBufferPointer{ result in
                        originalStrides.map{ Int32($0) }.withUnsafeBufferPointer{ osBuffer in
                            newStrides.map{ Int32($0) }.withUnsafeBufferPointer{ nsBuffer in
                                v.gradient!.withUnsafeBufferPointer{ xBuffer in
                                    output.shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                                        inverseOrder.map{ Int32($0) }.withUnsafeBufferPointer{ oBuffer in
                                            TKCore.permuteNoGradD(
                                                xBuffer.baseAddress! as? UnsafePointer<Double>,
                                                result.baseAddress! as? UnsafeMutablePointer<Double>,
                                                sBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                oBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                osBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                nsBuffer.baseAddress! as? UnsafePointer<Int32>,
                                                Int32(v.dataSize),
                                                Int32(v.shape.count)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                return result
            })
        ]
        output.operation = "Permute: \(order)"
        return output
    }
    /*
    @inlinable
    public func transpose(_ a: Int, _ b: Int) -> Tensor {
        guard a < shape.count, b < shape.count, a != b else {
            fatalError("Invalid indices \(a), \(b)")
        }
        var initialShape = shape
        initialShape.remove(at: a)
        initialShape.remove(at: b)
        if a < b {
            initialShape.insert(b, at: a)
            initialShape.insert(a, at: b)
        } else {
            initialShape.insert(a, at: b)
            initialShape.insert(b, at: a)
        }
        print("New Shape: \(initialShape)")
        return self.permute(initialShape)
    }*/
    
    
    @inlinable
    public func squeeze() -> Tensor {
        let result = Tensor(data, shape: self.shape.filter{ $0 != 1 }, calculate_grad: gradient != nil)
        result.parents = [
            (self, { v in v.gradient! })
        ]
        result.operation = "Squeeze"
        return result
    }
    
    @inlinable
    public func squeeze(_ dimension: Int) -> Tensor {
        guard shape[dimension] == 1 else {
            fatalError("Cannot squeeze along dimension \(dimension): dimension is not 1")
        }
        var newShape = shape
        newShape.remove(at: dimension)
        let result = Tensor(data, shape: newShape, calculate_grad: gradient != nil)
        result.parents = [
            (self, { v in v.gradient! })
        ]
        result.operation = "Squeeze"
        return result
    }
    
    @inlinable
    public func unsqueeze(_ dimension: Int) -> Tensor {
        let result = Tensor(data, shape: shape.inserting(1, at: dimension), calculate_grad: gradient != nil)
        result.parents = [
            (self, { v in v.gradient! })
        ]
        result.operation = "Unsqueeze"
        return result
    }
    
    @inlinable
    public func select(_ indices: [[Int]]) -> Tensor {
        var resultData = [T](repeating: 0, count: indices.count)
        for (index, i) in indices.enumerated() {
            resultData[index] = data[ravelIndex(i)]
        }
        let output = Tensor(resultData, shape: [indices.count], calculate_grad: self.gradient != nil)
        output.parents = [
            (self, { v in
                var resultData = [T](repeating: 0, count: self.gradient!.count)
                for (index, i) in indices.enumerated() {
                    resultData[self.ravelIndex(i)] = v.gradient![index]
                }
                return resultData
            })
        ]
        output.operation = "Select [\(indices.count)]"
        return output
    }
    
    @inlinable
    public func argmax(_ dim: Int) -> Tensor {
        let strides = generateStrides(shape)
        var outputShape = shape
        outputShape.remove(at: dim)
        outputShape.insert(1, at: dim)
        let numBlocks = dataSize / shape[dim]
        let blockStride = shape.dropFirst(dim + 1).reduce(1, *)
        if T.self == Float.self {
            var resultData = [Float](repeating: 0, count: numBlocks)
            data.withUnsafeBufferPointer{ xBuffer in
                shape.withUnsafeBufferPointer{ sBuffer in
                    TKCore.argmax(
                        xBuffer.baseAddress! as? UnsafePointer<Float>,
                        sBuffer.baseAddress! as? UnsafePointer<Int32>,
                        Int32(dim),
                        Int32(shape[dim]),
                        Int32(numBlocks),
                        Int32(blockStride),
                        &resultData
                    )
                }
            }
            return Tensor(resultData as! [T], shape: outputShape)
        } else if T.self == Double.self {
            var resultData = [Double](repeating: 0, count: numBlocks)
            data.withUnsafeBufferPointer{ xBuffer in
                shape.withUnsafeBufferPointer{ sBuffer in
                    TKCore.argmaxD(
                        xBuffer.baseAddress! as? UnsafePointer<Double>,
                        sBuffer.baseAddress! as? UnsafePointer<Int32>,
                        Int32(dim),
                        Int32(shape[dim]),
                        Int32(numBlocks),
                        Int32(blockStride),
                        &resultData
                    )
                }
            }
            return Tensor(resultData as! [T], shape: outputShape)
        }
        return Tensor(0.0)
    }
    
    @inlinable
    public func max(_ dim: Int) -> Tensor {
        var resultShape = shape
        resultShape.remove(at: dim)
        resultShape.insert(1, at: dim)
        let blockStride = shape.dropFirst(dim + 1).reduce(1, *)
        let dimSize = shape[dim]
        let numBlocks = dataSize / shape[dim]
        if T.self == Float.self {
            var resultData = [Float](repeating: 0, count: numBlocks)
            data.withUnsafeBufferPointer{ xBuffer in
                shape.withUnsafeBufferPointer{ sBuffer in
                    TKCore.max(
                        xBuffer.baseAddress! as? UnsafePointer<Float>,
                        sBuffer.baseAddress! as? UnsafePointer<Int32>,
                        Int32(dim),
                        Int32(dimSize),
                        Int32(numBlocks),
                        Int32(blockStride),
                        &resultData
                    )
                }
            }
            return Tensor(resultData as! [T], shape: resultShape)
        } else if T.self == Double.self {
            var resultData = [Double](repeating: 0, count: numBlocks)
            data.withUnsafeBufferPointer{ xBuffer in
                shape.withUnsafeBufferPointer{ sBuffer in
                    TKCore.maxD(
                        xBuffer.baseAddress! as? UnsafePointer<Double>,
                        sBuffer.baseAddress! as? UnsafePointer<Int32>,
                        Int32(dim),
                        Int32(dimSize),
                        Int32(numBlocks),
                        Int32(blockStride),
                        &resultData
                    )
                }
            }
            return Tensor(resultData as! [T], shape: resultShape)
        }
        return Tensor(0.0)
    }
    
    @inlinable
    public func min(_ dim: Int) -> Tensor {
        var resultShape = shape
        resultShape.remove(at: dim)
        resultShape.insert(1, at: dim)
        let blockStride = shape.dropFirst(dim + 1).reduce(1, *)
        let dimSize = shape[dim]
        let numBlocks = dataSize / shape[dim]
        if T.self == Float.self {
            var resultData = [Float](repeating: 0, count: numBlocks)
            data.withUnsafeBufferPointer{ xBuffer in
                shape.withUnsafeBufferPointer{ sBuffer in
                    TKCore.min(
                        xBuffer.baseAddress! as? UnsafePointer<Float>,
                        sBuffer.baseAddress! as? UnsafePointer<Int32>,
                        Int32(dim),
                        Int32(dimSize),
                        Int32(numBlocks),
                        Int32(blockStride),
                        &resultData
                    )
                }
            }
            return Tensor(resultData as! [T], shape: resultShape)
        } else if T.self == Double.self {
            var resultData = [Double](repeating: 0, count: numBlocks)
            data.withUnsafeBufferPointer{ xBuffer in
                shape.withUnsafeBufferPointer{ sBuffer in
                    TKCore.minD(
                        xBuffer.baseAddress! as? UnsafePointer<Double>,
                        sBuffer.baseAddress! as? UnsafePointer<Int32>,
                        Int32(dim),
                        Int32(dimSize),
                        Int32(numBlocks),
                        Int32(blockStride),
                        &resultData
                    )
                }
            }
            return Tensor(resultData as! [T], shape: resultShape)
        }
        return Tensor(0.0)
    }
    
    /*@inlinable
    public func toDoublePrecision() -> Tensor<Double> {
        guard T.self == Float.self else {
            if T.self == Double.self {
                return self as! Tensor<Double>
            } else {
                fatalError("This operation hasnt been implemented for \(T.self).")
            }
        }
        
        var outputData = [Double](repeating: 0, count: dataSize)
        data.withUnsafeBufferPointer{ iBuffer in
            outputData.withUnsafeMutableBufferPointer{ oBuffer in
                vDSP_vspdp(
                    iBuffer.baseAddress! as! UnsafePointer<Float>,
                    1,
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Double>,
                    1,
                    vDSP_Length(dataSize)
                )
            }
        }
        let result = Tensor<Double>(outputData, shape: shape, calculate_grad: gradient != nil)
        let inputCpy = result
        result.operation = "Convert \(T.self) to Double"
        result.parents = [
            (self, { v in
                
                data.withUnsafeBufferPointer{ iBuffer in
                    v.gradient!.withUnsafeMutableBufferPointer{ oBuffer in
                        vDSP_vspdp(
                            iBuffer.baseAddress! as! UnsafePointer<Float>,
                            1,
                            oBuffer.baseAddress! as! UnsafeMutablePointer<Double>,
                            1,
                            vDSP_Length(dataSize)
                        )
                    }
                }
            })
        ]
    }*/
    
    @inlinable
    public func isolate(_ dim: Int, _ index: Int, keepDims: Bool = true) -> Tensor<T> {
        guard dim < shape.count && index < shape[dim] else {
            fatalError("Cannot isolate data on dimension \(dim), index \(index). Tensor is shape: \(shape).")
        }
        
        let strides = generateStrides(shape)
        let dimStride = strides[dim]
        let numValues = dataSize / shape[dim]
        var result = [T](repeating: 0, count: numValues)
        var multiIndex = [Int](repeating: 0, count: shape.count)

        for i in 0..<numValues {
                var flatIdx = 0
                var idxInResult = i
                for d in stride(from: shape.count - 1, through: 0, by: -1) {
                    if d == dim {
                        multiIndex[d] = index
                    } else {
                        multiIndex[d] = idxInResult % shape[d]
                        idxInResult /= shape[d]
                    }
                    flatIdx += multiIndex[d] * strides[d]
                }
                result[i] = data[flatIdx]
            }

        var outputShape: [Int] = []
        if keepDims {
            outputShape = shape
            outputShape.remove(at: dim)
            outputShape.insert(1, at: dim)
        } else {
            outputShape = [numValues]
        }

        let output = Tensor(result, shape: outputShape)
        output.operation = "Isolate [Dimension: \(dim), Index: \(index)]"
        return output
    }
    
    @inlinable
    public func shuffle(_ dim: Int?) -> Tensor<T> {
        if dim != nil {
            guard dim! < shape.count else {
                fatalError("Cannot shuffle along dimension \(dim!); shape is: \(shape)")
            }
        }
        
        guard let dim = dim else {
            let shuffleMap = intRamp(dataSize).shuffled()
            let shuffled = shuffleMap.map{ data[$0] }
            let output = Tensor(shuffled, shape: shape, calculate_grad: gradient != nil)
            if gradient != nil {
                output.parents = [
                    (self, { v in
                        var inverseIndexMap = Array(repeating: 0, count: shuffleMap.count)
                        for (shuffledIndex, originalIndex) in shuffleMap.enumerated() {
                            inverseIndexMap[originalIndex] = shuffledIndex
                        }
                        return inverseIndexMap.map{ v.gradient![$0] }
                    })
                ]
            }
            output.operation = "Shuffle [All]"
            return output
        }
        
        let numApplied = dim == 0 ? 1 : shape[0..<dim].reduce(1, *)
        //var shuffleMap = [Int](repeating: 0, count: shape[0...dim].reduce(1, *))
        /*let stride = generateStrides(shape)[dim]
        let resultData = [T](repeating: 0, count: dataSize)
        for i in 0..<numApplied {
            let shuffleMap = intRamp(shape[dim])
            let outerStride = numApplied
            for j in 0..<shape[dim] {
                resultData[j * stride..<(j + 1) * stride] =
            }
        }*/
        var resultData = data
        let indices = intRamp(shape[dim]).shuffled()
        let strides = generateStrides(shape)
        for flattenedIndex in 0..<dataSize {
            var multiIndex = unflattenedIndex(flattenedIndex, strides: shape, shape: shape)
            multiIndex[dim] = indices[multiIndex[dim]]
            let targetFlatindex = flatIndex(strides: strides, index: multiIndex)
            resultData[targetFlatindex] = data[flattenedIndex]
        }
        return Tensor(resultData, shape: shape)
    }
    
    @inlinable
    public func clearGrad() {
        if gradient != nil {
            if T.self == Float.self {
                gradient!.withUnsafeMutableBufferPointer{ grad in
                    vDSP_vclr(grad.baseAddress! as! UnsafeMutablePointer<Float>, 1, vDSP_Length(dataSize))
                }
            } else if T.self == Double.self {
                gradient!.withUnsafeMutableBufferPointer{ grad in
                    vDSP_vclrD(grad.baseAddress! as! UnsafeMutablePointer<Double>, 1, vDSP_Length(dataSize))
                }
            } else {
                gradient = [T](repeating: 0, count: data.count)
            }
        }
    }
    
    @inlinable
    public func disown() {
        parents = []
    }
    
    @inlinable
    public func backward(_ grad: [T] = Array(repeating: 1.0, count: 0), printSteps: Bool = false, printParams: Bool = false) {
        if parents.count != 0 || gradient != nil {
            if gradient == nil {
                gradient = [T](repeating: 0, count: data.count)
            }
            // Check if the gradient sizes match
            let grad = grad.isEmpty ? [T](repeating: 1.0, count: gradient!.count) : grad
            // Accumulate gradients
            gradient = add(gradient!, grad)
            if printSteps {
                print("Grad At Operation: \(operation ?? "Leaf")")
                print(Tensor(gradient!, shape: shape))
            }
            if printParams {
                if operation != nil {
                    if operation!.contains("Weight") || operation!.contains("Bias") {
                        print(operation!)
                        print(Tensor(gradient!, shape: shape))
                    }
                }
            }
            if operation != nil {
                if operation!.contains("FG") {
                    print("Gradient of Operation \(operation!):")
                    print(Tensor(gradient!, shape: shape))
                }
            }
        }
        for (parent, local) in parents {
            guard parent.parents.count != 0 || parent.gradient != nil else {
                continue
            }
            let localGradients = local(self)
            parent.backward(localGradients, printSteps: printSteps, printParams: printParams)
        }
        if parents.count != 0 {
            if printSteps || printParams {
                clearGrad()
            } else {
                gradient = nil
            }
        }
    }
}

@inlinable
public func ramp<T: TensorComplex>(_ count: Int) -> [T] {
    if T.self == Float.self  {
        var start: Float = 0
        var end = Float(count - 1)
        let result = [Float](unsafeUninitializedCapacity: count) { buffer, initializedCount in
            vDSP_vgen(
                &start,
                &end,
                buffer.baseAddress!,
                1,
                vDSP_Length(count)
            )
            initializedCount = count
        }
        return result as! [T]
    } else if T.self == Double.self {
        var start: Double = 0
        var end = Double(count - 1)
        let result = [Double](unsafeUninitializedCapacity: count) { buffer, initializedCount in
            vDSP_vgenD(
                &start,
                &end,
                buffer.baseAddress!,
                1,
                vDSP_Length(count)
            )
            initializedCount = count
        }
        return result as! [T]
    }
    return []
}

@inlinable
public func intRamp(_ count: Int) -> [Int] {
    var start: Float = 0
    var end = Float(count - 1)
    let result = [Float](unsafeUninitializedCapacity: count) { buffer, initializedCount in
        vDSP_vgen(
            &start,
            &end,
            buffer.baseAddress!,
            1,
            vDSP_Length(count)
        )
        initializedCount = count
    }
    return result.map{ Int($0) }
}

@inlinable
public func upperTriangle<T: TensorComplex>(rows: Int, cols: Int, upper: T, lower: T) -> [T] {
    var result = [T](repeating: 0, count: rows * cols)
    if T.self == Float.self {
        result.withUnsafeMutableBufferPointer { oBuffer in
            upperTriangle(
                Int32(rows),
                Int32(cols),
                upper as! Float,
                lower as! Float,
                oBuffer.baseAddress! as? UnsafeMutablePointer<Float>
            )
        }
        return result
    } else if T.self == Double.self {
        result.withUnsafeMutableBufferPointer { oBuffer in
            upperTriangleD(
                Int32(rows),
                Int32(cols),
                upper as! Double,
                lower as! Double,
                oBuffer.baseAddress! as? UnsafeMutablePointer<Double>
            )
        }
        return result
    } else {
        return result
    }
}

@inlinable
public func lowerTriangle<T: TensorComplex>(rows: Int, cols: Int, upper: T, lower: T) -> [T] {
    var result = [T](repeating: 0, count: rows * cols)
    if T.self == Float.self {
        result.withUnsafeMutableBufferPointer { oBuffer in
            lowerTriangle(
                Int32(rows),
                Int32(cols),
                upper as! Float,
                lower as! Float,
                oBuffer.baseAddress! as? UnsafeMutablePointer<Float>
            )
        }
        return result
    } else if T.self == Double.self {
        result.withUnsafeMutableBufferPointer { oBuffer in
            lowerTriangleD(
                Int32(rows),
                Int32(cols),
                upper as! Double,
                lower as! Double,
                oBuffer.baseAddress! as? UnsafeMutablePointer<Double>
            )
        }
        return result
    } else {
        return result
    }
}

@inlinable
public func upperTriangle<T: TensorComplex>(shape: [Int], upper: T, lower: T) -> Tensor<T> {
    let count = shape.reduce(1, *) / shape.last! / shape.dropLast().last!
    var result = [T]()
    for _ in 0..<count {
        result.append(contentsOf: upperTriangle(rows: shape.dropLast().last!, cols: shape.last!, upper: upper, lower: lower))
    }
    return Tensor(result, shape: shape)
}

@inlinable
public func lowerTriangle<T: TensorComplex>(shape: [Int], upper: T, lower: T) -> Tensor<T> {
    let count = shape.reduce(1, *) / shape.last! / shape.dropLast().last!
    var result = [T]()
    for _ in 0..<count {
        result.append(contentsOf: lowerTriangle(rows: shape.dropLast().last!, cols: shape.last!, upper: upper, lower: lower))
    }
    return Tensor(result, shape: shape)
}

@inlinable
public func selectShape(from shape: [Int], using indices: [Int]) -> [Int] {
    var selectedShape: [Int] = []
    for index in indices {
        if index >= 0 && index < shape.count {
            selectedShape.append(shape[index])
        }
    }
    return selectedShape
}
@inlinable
public func generateStrides(_ shape: [Int]) -> [Int] {
    var result = [Int](repeating: 1, count: shape.count)
    var stride = 1
    for i in (0..<shape.count).reversed() {
        result[i] = stride
        stride *= shape[i]
    }
    return result
}
@inlinable
public func flatIndex(strides: [Int], index: [Int]) -> Int {
    let a = zip(strides, index).map(*)
    return a.reduce(0, +)
}

@inlinable
public func unflattenedIndex(_ index: Int, strides: [Int], shape: [Int]) -> [Int] {
    var multiIndex = [Int](repeating: 0, count: shape.count)
    var remainder = index
        for i in 0..<shape.count {
            multiIndex[i] = remainder / strides[i]
            remainder %= strides[i]
        }
    return multiIndex
}
@inlinable
public func sum<T: TensorComplex>(_ input: [T], shape: [Int], along: Int) -> [T] {
    // Ensure the dimension is valid
    precondition(along < shape.count, "Invalid dimension for sum")

    let dimSize = shape[along]
    var summedValues = [T](repeating: 0.0, count: input.count / dimSize)

    // Calculate the stride and number of blocks for the specified dimension
    let numBlocks = input.count / dimSize
    let blockStride = shape.dropFirst(along + 1).reduce(1, *)
    if T.self == Float.self {
        input.withUnsafeBufferPointer{ iBuffer in
            summedValues.withUnsafeMutableBufferPointer{ yBuffer in
                shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                    TKCore.sum(
                        iBuffer.baseAddress! as? UnsafePointer<Float>,
                        Int32(along),
                        yBuffer.baseAddress! as? UnsafeMutablePointer<Float>,
                        sBuffer.baseAddress! as? UnsafePointer<Int32>,
                        Int32(0),
                        Int32(numBlocks),
                        Int32(blockStride)
                    )
                }
            }
        }
    } else if T.self == Double.self {
        input.withUnsafeBufferPointer{ iBuffer in
            summedValues.withUnsafeMutableBufferPointer{ yBuffer in
                shape.map{ Int32($0) }.withUnsafeBufferPointer{ sBuffer in
                    TKCore.sumD(
                        iBuffer.baseAddress! as? UnsafePointer<Double>,
                        Int32(along),
                        yBuffer.baseAddress! as? UnsafeMutablePointer<Double>,
                        sBuffer.baseAddress! as? UnsafePointer<Int32>,
                        Int32(0),
                        Int32(numBlocks),
                        Int32(blockStride)
                    )
                }
            }
        }
    }
/*
    // Perform the summation along the specified dimension
    for blockIndex in 0..<numBlocks {
        // Compute the starting index for this block
        let blockStartIndex = (blockIndex / blockStride) * (blockStride * dimSize) + (blockIndex % blockStride)

        for j in 0..<dimSize {
            let index = blockStartIndex + j * blockStride
            summedValues[blockIndex] += input[index]
        }
    }*/

    return summedValues
}


// Helper function to calculate the flattened index
@inlinable
public func flattenedIndex(of index: Int, withShape newShape: [Int], along dimension: Int, innerIndex: Int, originalShape: [Int]) -> Int {
    var indices = [Int](repeating: 0, count: originalShape.count)
    
    // Determine the indices in the original shape
    var remainingIndex = index
    for d in (0..<originalShape.count).reversed() {
        if d == dimension {
            indices[d] = innerIndex
        } else {
            let size = originalShape[d]
            indices[d] = remainingIndex % size
            remainingIndex /= size
        }
    }
    
    // Flatten the indices to a single index
    return indices.enumerated().reduce(0) { acc, pair in
        acc + pair.element * originalShape[pair.offset]
    }
}

@inlinable
public func sum<T: TensorComplex>(_ data: [T], shape: [Int], along: [Int]) -> [T] {
    var result = data
    for i in along {
        result = sum(result, shape: shape, along: i)
    }
    return result
}

@inlinable
public func sum<T: TensorComplex>(_ data: [T], shape: [Int], to: [Int]) -> [T] {
    var result = data
    var toShape = to
    while toShape.count < shape.count {
        toShape.insert(1, at: 0)
    }
    for i in 0..<to.count {
        if toShape[i] == 1 && shape[i] != 1 {
            result = sum(result, shape: shape, along: i)
        }
    }
    
    return result
}

@inlinable
public func sum<T: TensorComplex>(_ data: [T], shape: [Int]) -> T {
    var result = data
    var toShape = [T](repeating: 1, count: shape.count)
    while toShape.count < shape.count {
        toShape.insert(1, at: 0)
    }
    for i in 0..<shape.count {
        if shape[i] != 1 {
            result = sum(result, shape: shape, along: i)
        }
    }
    
    return result[0]
}

@inlinable
public func mergeShapes(_ a: [Int], _ b: [Int]) -> [Int] {
    guard a != b else {
        return a
    }
    if a.reduce(1, *) > b.reduce(1, *) {
        return a
    } else if a.reduce(1, *) == b.reduce(1, *) {
        return a.reduce(1, +) > b.reduce(1, +) ? a : b
    } else {
        return b
    }
}

@inlinable
public func multiply<T: TensorComplex>(_ input: [T], s: T) -> [T] {
    var result = [T]()
    let totalElements = input.count
    
    if T.self == Float.self {
        var outputData = [T](repeating: 0, count: totalElements)
        input.withUnsafeBufferPointer { rBuffer in
            outputData.withUnsafeMutableBufferPointer { oBuffer in
                vDSP_vsmul(rBuffer.baseAddress! as! UnsafePointer<Float>,
                    1,
                    [s] as! [Float],
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                    1,
                    vDSP_Length(totalElements)
                )
            }
        }
        
        result = outputData
    } else if T.self == Double.self {
        var outputData = [T](repeating: 0, count: totalElements)
        input.withUnsafeBufferPointer { rBuffer in
            outputData.withUnsafeMutableBufferPointer { oBuffer in
                vDSP_vsmulD(rBuffer.baseAddress! as! UnsafePointer<Double>,
                    1,
                    [s] as! [Double],
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Double>,
                    1,
                    vDSP_Length(totalElements)
                )
            }
        }
        
        result = outputData
    } else {
        var outputData = [Float](repeating: 0, count: totalElements)
        let lDataFloat = input.compactMap { Float($0) }
        lDataFloat.withUnsafeBufferPointer { rBuffer in
            outputData.withUnsafeMutableBufferPointer { oBuffer in
                vDSP_vsmul(rBuffer.baseAddress! as! UnsafePointer<Float>,
                    1,
                    [s] as! [Float],
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                    1,
                    vDSP_Length(totalElements)
                )
            }
        }
        result = outputData.compactMap{ T($0) }
    }
    return result
}

@inlinable
public func divide<T: TensorComplex>(_ input: [T], s: T) -> [T] {
    var result = [T]()
    let totalElements = input.count
    
    if T.self == Float.self {
        var outputData = [T](repeating: 0, count: totalElements)
        input.withUnsafeBufferPointer { rBuffer in
            outputData.withUnsafeMutableBufferPointer { oBuffer in
                vDSP_vsdiv(
                    rBuffer.baseAddress! as! UnsafePointer<Float>,
                    1,
                    [s] as! [Float],
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                    1,
                    vDSP_Length(totalElements)
                )
            }
        }
        
        result = outputData
    } else if T.self == Double.self {
        var outputData = [T](repeating: 0, count: totalElements)
        input.withUnsafeBufferPointer { rBuffer in
            outputData.withUnsafeMutableBufferPointer { oBuffer in
                vDSP_vsdivD(
                    rBuffer.baseAddress! as! UnsafePointer<Double>,
                    1,
                    [s] as! [Double],
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Double>,
                    1,
                    vDSP_Length(totalElements)
                )
            }
        }
        
        result = outputData
    } else {
        var outputData = [Float](repeating: 0, count: totalElements)
        let lDataFloat = input.compactMap { Float($0) }
        lDataFloat.withUnsafeBufferPointer { rBuffer in
            outputData.withUnsafeMutableBufferPointer { oBuffer in
                vDSP_vsdiv(
                    rBuffer.baseAddress! as! UnsafePointer<Float>,
                    1,
                    [s] as! [Float],
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                    1,
                    vDSP_Length(totalElements)
                )
            }
        }
    }
    return result
}

@inlinable
public func inverseDivide<T: TensorComplex>(_ input: [T], s: T) -> [T] {
    var result = [T]()
    let totalElements = input.count
    
    if T.self == Float.self {
        var outputData = [T](repeating: 0, count: totalElements)
        input.withUnsafeBufferPointer { rBuffer in
            outputData.withUnsafeMutableBufferPointer { oBuffer in
                vDSP_svdiv(
                    [s] as! [Float],
                    rBuffer.baseAddress! as! UnsafePointer<Float>,
                    1,
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                    1,
                    vDSP_Length(totalElements)
                )
            }
        }
        
        result = outputData
    } else if T.self == Double.self {
        var outputData = [T](repeating: 0, count: totalElements)
        input.withUnsafeBufferPointer { rBuffer in
            outputData.withUnsafeMutableBufferPointer { oBuffer in
                vDSP_svdivD(
                    [s] as! [Double],
                    rBuffer.baseAddress! as! UnsafePointer<Double>,
                    1,
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Double>,
                    1,
                    vDSP_Length(totalElements)
                )
            }
        }
        
        result = outputData
    } else {
        var outputData = [Float](repeating: 0, count: totalElements)
        let lDataFloat = input.compactMap { Float($0) }
        lDataFloat.withUnsafeBufferPointer { rBuffer in
            outputData.withUnsafeMutableBufferPointer { oBuffer in
                vDSP_svdiv(
                    [s] as! [Float],
                    rBuffer.baseAddress! as! UnsafePointer<Float>,
                    1,
                    oBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                    1,
                    vDSP_Length(totalElements)
                )
            }
        }
    }
    return result
}

@inlinable
public func matrixMultiply<T: TensorComplex>(_ a: [T], _ b: [T], aShape: [Int], bShape: [Int]) -> [T] {
    guard aShape.count >= 2 && bShape.count >= 2 else {
        fatalError("Not enough dimensions for matrix multiplication")
    }
    guard aShape.last! == bShape.dropLast().last! else {
        fatalError("Incorrect shapes for matrix multiplication")
    }
    var finalShape = mergeShapes(aShape, bShape)
    finalShape[finalShape.endIndex - 1] = bShape.last!
    finalShape[finalShape.endIndex - 2] = aShape.dropLast().last!
    let aRows = aShape.dropLast().last!
    let aCols = aShape.last!
    let bCols = bShape.last!
    let outputSize = finalShape.reduce(1, *)
    let batchCount = aShape.dropLast(2).reduce(1, *)
    var result = [T]()

        if T.self == Float.self {
            var outputData = [T](repeating: 0, count: outputSize)
            for batch in 0..<batchCount {
                let lOffset = batch * aRows * aCols
                let rOffset = batch * aCols * bCols
                let oOffset = batch * aRows * bCols
                a.withUnsafeBufferPointer { lBuffer in
                    b.withUnsafeBufferPointer { rBuffer in
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
            result = outputData
        } else if T.self == Double.self {
            var outputData = [T](repeating: 0, count: outputSize)
            for batch in 0..<batchCount {
                let lOffset = batch * aRows * aCols
                let rOffset = batch * aCols * bCols
                let oOffset = batch * aRows * bCols
                a.withUnsafeBufferPointer { lBuffer in
                    b.withUnsafeBufferPointer { rBuffer in
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
            result = outputData
        } else {
            var outputData = [Float](repeating: 0, count: outputSize)
            
            let lDataFloat = a.compactMap { Float($0) }
            let rDataFloat = b.compactMap { Float($0) }
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
            result = outputData.compactMap{ T($0) }
        }
    return result
}

@inlinable
public func transpose<T: TensorComplex>(_ input: [T], shape: [Int]) -> [T] {
    var result = [T]()
    let rows = shape.dropLast().last!
    let cols = shape.last!
    let totalElements = input.count
    let batchCount = shape.dropLast(2).reduce(1, *)
    
    if T.self == Float.self {
        var outputData = [T](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            input.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtrans(sdata.baseAddress! as! UnsafePointer<Float> + offset, 1, odata.baseAddress! as! UnsafeMutablePointer<Float> + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData
    } else if T.self == Double.self {
        var outputData = [T](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            input.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtransD(sdata.baseAddress! as! UnsafePointer<Double> + offset, 1, odata.baseAddress! as! UnsafeMutablePointer<Double> + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData
    } else {
        let compData = input.compactMap{ Float($0) }
        var outputData = [Float](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            compData.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtrans(sdata.baseAddress! + offset, 1, odata.baseAddress! + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData.compactMap{ T($0) }
    }
    
    return result
}

@inlinable
public func transpose<T: TensorComplex>(_ input: Tensor<T>) -> Tensor<T> {
    var result = [T]()
    let rows = input.shape.dropLast().last!
    let cols = input.shape.last!
    let totalElements = input.data.count
    let batchCount = input.shape.dropLast(2).reduce(1, *)
    
    if T.self == Float.self {
        var outputData = [T](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            input.data.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtrans(sdata.baseAddress! as! UnsafePointer<Float> + offset, 1, odata.baseAddress! as! UnsafeMutablePointer<Float> + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData
    } else if T.self == Double.self {
        var outputData = [T](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            input.data.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtransD(sdata.baseAddress! as! UnsafePointer<Double> + offset, 1, odata.baseAddress! as! UnsafeMutablePointer<Double> + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData
    } else {
        let compData = input.data.compactMap{ Float($0) }
        var outputData = [Float](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            compData.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtrans(sdata.baseAddress! + offset, 1, odata.baseAddress! + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData.compactMap{ T($0) }
    }
    var outputshape = input.shape
    outputshape.swapAt(outputshape.count - 2, outputshape.count - 1)
    let output = Tensor(result, shape: outputshape, calculate_grad: input.gradient != nil)
    output.operation = "Transpose - Global"
    output.parents = [
        (input, { v in transpose(v.gradient!, shape: v.shape)})
    ]
    return output
}

@inlinable
public func add<T: TensorComplex>(_ x: [T], _ y: [T]) -> [T] {
    guard x.count == y.count else {
        fatalError("Mismatching inputs to add function: \(x.count) & \(y.count)")
    }
    let result = x.count
    if T.self == Float.self {
        var outputData = [T](repeating: 0, count: x.count)
        
        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vadd(
                        lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                        vDSP_Length(result)
                    )
                }
            }
        }
        
        return outputData
    } else if T.self == Double.self {
        var outputData = [T](repeating: 0, count: result)

        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vaddD(
                        lBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                        rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                        oBuffer.baseAddress! as! UnsafeMutablePointer<Double>, 1,
                        vDSP_Length(result)
                    )
                }
            }
        }
        
        return outputData
    } else {
        var outputData = [Float](repeating: 0, count: result)
        
        let lDataFloat = x.compactMap { Float($0) }
        let rDataFloat = y.compactMap { Float($0) }
        
        lDataFloat.withUnsafeBufferPointer { lBuffer in
            rDataFloat.withUnsafeBufferPointer { rBuffer in
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
    }
}

@inlinable
public func multiply<T: TensorComplex>(_ x: [T], _ y: [T]) -> [T] {
    guard x.count == y.count else {
        fatalError("Mismatching inputs to multiply() function: \(x.count) & \(y.count)")
    }
    let result = x.count
    if T.self == Float.self {
        var outputData = [T](repeating: 0, count: x.count)
        
        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vmul(
                        lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                        vDSP_Length(result)
                    )
                }
            }
        }
        
        return outputData
    } else if T.self == Double.self {
        var outputData = [T](repeating: 0, count: result)
        
        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vmulD(
                        lBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                        rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                        oBuffer.baseAddress! as! UnsafeMutablePointer<Double>, 1,
                        vDSP_Length(result)
                    )
                }
            }
        }
        
        return outputData
    } else {
        var outputData = [Float](repeating: 0, count: result)
        
        let lDataFloat = x.compactMap { Float($0) }
        let rDataFloat = y.compactMap { Float($0) }
        
        lDataFloat.withUnsafeBufferPointer { lBuffer in
            rDataFloat.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vmul(
                        lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                        vDSP_Length(result)
                    )
                }
            }
        }
        return outputData as! [T]
    }
}

@inlinable
public func divide<T: TensorComplex>(_ x: [T], _ y: [T]) -> [T] {
    guard x.count == y.count else {
        fatalError("Mismatching inputs to multiply() function: \(x.count) & \(y.count)")
    }
    let result = x.count
    if T.self == Float.self {
        var outputData = [T](repeating: 0, count: x.count)
        
        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vdiv(
                        lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                        vDSP_Length(result)
                    )
                }
            }
        }
        
        return outputData
    } else if T.self == Double.self {
        var outputData = [T](repeating: 0, count: result)
        
        x.withUnsafeBufferPointer { lBuffer in
            y.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vdivD(
                        lBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                        rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                        oBuffer.baseAddress! as! UnsafeMutablePointer<Double>, 1,
                        vDSP_Length(result)
                    )
                }
            }
        }
        
        return outputData
    } else {
        var outputData = [Float](repeating: 0, count: result)
        
        let lDataFloat = x.compactMap { Float($0) }
        let rDataFloat = y.compactMap { Float($0) }
        
        lDataFloat.withUnsafeBufferPointer { lBuffer in
            rDataFloat.withUnsafeBufferPointer { rBuffer in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vdiv(
                        lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                        vDSP_Length(result)
                    )
                }
            }
        }
        return outputData as! [T]
    }
}

@inlinable
public func sin<T: TensorComplex>(_ x: [T]) -> [T] {
    var totalSize = Int32(x.count)
    var result = [T](repeating: 0, count: x.count)
    if T.self == Float.self {
        x.withUnsafeBufferPointer { xBuffer in
            result.withUnsafeMutableBufferPointer { yBuffer in
                vvsinf(yBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                       xBuffer.baseAddress! as! UnsafePointer<Float>,
                       &totalSize
                )
            }
        }
        return result
    } else if T.self == Double.self {
        x.withUnsafeBufferPointer { xBuffer in
            result.withUnsafeMutableBufferPointer { yBuffer in
                vvsin(yBuffer.baseAddress! as! UnsafeMutablePointer<Double>,
                       xBuffer.baseAddress! as! UnsafePointer<Double>,
                       &totalSize
                )
            }
        }
        return result
    } else {
        x.withUnsafeBufferPointer { xBuffer in
            result.withUnsafeMutableBufferPointer { yBuffer in
                vvsinf(yBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                       xBuffer.baseAddress! as! UnsafePointer<Float>,
                       &totalSize
                )
            }
        }
        return result
    }
}

@inlinable
public func cos<T: TensorComplex>(_ x: [T]) -> [T] {
    var totalSize = Int32(x.count)
    var result = [T](repeating: 0, count: x.count)
    if T.self == Float.self {
        x.withUnsafeBufferPointer { xBuffer in
            result.withUnsafeMutableBufferPointer { yBuffer in
                vvcosf(yBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                       xBuffer.baseAddress! as! UnsafePointer<Float>,
                       &totalSize
                )
            }
        }
        return result
    } else if T.self == Double.self {
        x.withUnsafeBufferPointer { xBuffer in
            result.withUnsafeMutableBufferPointer { yBuffer in
                vvcos(yBuffer.baseAddress! as! UnsafeMutablePointer<Double>,
                       xBuffer.baseAddress! as! UnsafePointer<Double>,
                       &totalSize
                )
            }
        }
        return result
    } else {
        x.withUnsafeBufferPointer { xBuffer in
            result.withUnsafeMutableBufferPointer { yBuffer in
                vvcosf(yBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                       xBuffer.baseAddress! as! UnsafePointer<Float>,
                       &totalSize
                )
            }
        }
        return result
    }
}

@inlinable
public func concatenate<T: TensorComplex>(_ x: [Tensor<T>], dimension: Int) -> Tensor<T> {
    var totalLength = 0
    var comp = x[0].shape
    comp.remove(at: dimension)
    var parents: [(Tensor<T>, (Tensor<T>) -> [T])] = []
    var jSizes: [Int] = []
    var hasGrad = false
    var placementTensors: [Tensor<T>] = []
    for i in x {
        if i.gradient != nil {
            hasGrad = true
        }
        jSizes.append(i.shape[dimension])
        totalLength += i.shape[dimension]
        var reducedShape = i.shape
        reducedShape.remove(at: dimension)
        if reducedShape != comp {
            fatalError("Tensors cannot be concatenated along dimension \(dimension) due to size mismatch.")
        }
    }
    var resultShape = x[0].shape
    resultShape.remove(at: dimension)
    resultShape.insert(totalLength, at: dimension)
    let finalDataSize = resultShape.reduce(1, *)
    var parentIndices = [Int](repeating: 0, count: finalDataSize)
    let numBlocks = x[0].dataSize / x[0].shape[dimension]
    var result = [T](repeating: 0, count: finalDataSize)
    var gradResult = [T](repeating: 0, count: hasGrad ? finalDataSize : 0)
    let blockStride = x[0].shape.dropFirst(dimension + 1).reduce(1, *)
    for blockIndex in 0..<numBlocks {
        var slice = [T](repeating: 0, count: totalLength)
        var gslice = [T](repeating: 0, count: hasGrad ? totalLength : 0)
        var pislice = [Int](repeating: 0, count: hasGrad ? totalLength : 0)
        
        var sliceIndex = 0
        for (tindex, tensor) in x.enumerated() {
            let jSize = tensor.shape[dimension]
            let blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride)
            for i in 0..<jSize {
                let index = blockStartIndex + i * blockStride
                slice[sliceIndex] = tensor.data[index]
                if hasGrad {
                    gslice[sliceIndex] = tensor.gradient![index]
                    pislice[sliceIndex] = tindex
                }
                sliceIndex += 1
            }
        }
        let jSize = totalLength
        let blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride)
        for i in 0..<jSize {
            let index = blockStartIndex + i * blockStride
            result[index] = slice[i]
            if hasGrad {
                gradResult[index] = gslice[i]
                parentIndices[index] = pislice[i]
            }
        }
    }
    
    if hasGrad {
        for (outerIndex, i) in x.enumerated() {
            var originalShape = i.shape
            let placement = Tensor<T>(i.data, shape: i.shape)
            placement.parents = [
                (i, { v in
                    var result = [T](repeating: 0, count: i.dataSize)
                    var resultIndex = 0
                    for i in 0..<v.dataSize {
                        if parentIndices[i] == outerIndex {
                            result[resultIndex] = v.gradient![i]
                            resultIndex += 1
                        }
                    }
                    return result
                })
            ]
            parents.append(contentsOf: placement.parents)
        }
    }
    let output = Tensor<T>(result, shape: resultShape)
    output.parents = parents
    output.operation = "Concatenate"
    output.gradient = hasGrad ? gradResult : nil
    return output
}
