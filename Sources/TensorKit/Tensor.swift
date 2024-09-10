//
//  nTensor.swift
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

public class Tensor<T: TensorType>: Codable, CustomStringConvertible {
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
            self.data = [T](repeating: T(Double.random(in: -xavierScale...xavierScale)), count: shape.reduce(1, *))
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
    public func encode(to encoder: any Encoder) throws {
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
        self.gradient = try container.decode([T].self, forKey: .gradient)
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

    
    
    // Sum over a specified dimension
    @inlinable
    public func sum(along dimension: Int) -> Tensor {
        /*// Ensure the dimension is valid
        precondition(dimension < shape.count, "Invalid dimension for summation")
        
        var newDimensions = shape
        newDimensions.remove(at: dimension)
        
        let stride = shape[dimension]
        let newSize = data.count / stride
        var summedData = [T](repeating: 0.0, count: newSize)
        
        for i in 0..<newSize {
            for j in 0..<stride {
                let index = i + j * newSize
                summedData[i] += data[index]
            }
        } */
        
        var newDimensions = shape
        //newDimensions.remove(at: dimension)
        newDimensions[dimension] = 1
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
        result.operation = "sum"
        result.parents = [
            (self, { v in
                Tensor(v.gradient!, shape: v.shape).expand(to: self.shape).data
                })
        ]
        return result
    }
    
    @inlinable
    public func sum(along: [Int]) -> Tensor {
        var result = self
        for i in along {
            result = result.sum(along: i)
        }
        return result
    }
    /*public func sum() -> Tensor {
        var result = self
        for i in 0..<self.shape.count {
            if result.shape[i] != 1 {
                result = result.sum(along: i)
            }
        }
        return result
    }*/
    @inlinable
    public func sum() -> Tensor {
        var result = self
        result.shape = [1]
        if T.self == Float.self {
            data.withUnsafeBufferPointer { ibuffer in
                result.data.withUnsafeMutableBufferPointer{ rbuffer in
                    vDSP_sve(ibuffer.baseAddress! as! UnsafePointer<Float>, 1, rbuffer.baseAddress! as! UnsafeMutablePointer<Float>, vDSP_Length(dataSize))
                }
            }
            result.operation = "Sum"
            result.parents = [
                (self, { v in Tensor(v.gradient!, shape: result.shape).expand(to: self.shape).data })
            ]
            return result
        } else if T.self == Double.self {
            data.withUnsafeBufferPointer { ibuffer in
                result.data.withUnsafeMutableBufferPointer{ rbuffer in
                    vDSP_sveD(ibuffer.baseAddress! as! UnsafePointer<Double>, 1, rbuffer.baseAddress! as! UnsafeMutablePointer<Double>, vDSP_Length(dataSize))
                }
            }
            result.operation = "Sum"
            result.parents = [
                (self, { v in Tensor(v.gradient!, shape: result.shape).expand(to: self.shape).data })
            ]
            return result
        } else {
            for i in 0..<self.shape.count {
                if result.shape[i] != 1 {
                    result = result.sum(along: i)
                }
            }
            result.operation = "Sum"
            result.parents = [
                (self, { v in Tensor(v.gradient!, shape: result.shape).expand(to: self.shape).data })
            ]
            return result
        }
    }
    
    // Reshape to new dimensions
    /*func reshape(to newDimensions: [Int]) -> Tensor {
        // Ensure the total number of elements is consistent
        let totalElements = shape.reduce(1, *)
        let newTotalElements = newDimensions.reduce(1, *)
        if totalElements != newTotalElements {
            print("Current Shape: \(shape) To: \(newDimensions) With Total amounts: Current: \(totalElements) To: \(newTotalElements)")
        }
        precondition(totalElements == newTotalElements, "Total elements must remain the same for reshape")
        
        return Tensor(data, shape: newDimensions)
    }*/
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
        
        for i in (0..<newDimensions.count).reversed() {
            if newDimensions[i] == 1 && targetDimensions[i] > 1 {
                if i == newDimensions.count - 1 {
                    let returnCount = targetDimensions[i] - newDimensions[i]
                    let count = dataSize / newDimensions.last!
                    for row in 0..<count {
                        broadcastedData.insert(contentsOf: Array(repeating: data[row], count: returnCount), at: row * targetDimensions[i])
                    }
                } else {
                    let returnCount = targetDimensions[i] - newDimensions[i]
                    broadcastedData.append(contentsOf: repeatArray(broadcastedData, count: returnCount))
                    newDimensions[i] = targetDimensions[i]
                }
            }
        }
        let result = Tensor(broadcastedData, shape: targetDimensions, calculate_grad: gradient != nil)
        result.operation = "<->"
        result.parents = [
            (self, { v in TensorKit.sum(v.gradient!, shape: v.shape, along: gradientMap)})
        ]
        return result
    }
    
    @inlinable
    public func testingSpeed(to targetDimensions: [Int]) -> Tensor {
        guard targetDimensions != shape else {
            return self
        }
        
        func canExpand(_ a: [Int], _ b: [Int]) -> Bool {
            for (x, y) in zip(a, b) {
                if x == y || x == 1 || y == 1 {
                    continue
                } else {
                    return false
                }
            }
            return true
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
        
        guard canExpand(newDimensions, targetDimensions) else {
            fatalError("Could not expand tensor from \(newDimensions) to \(targetDimensions)")
        }
        
        let gradientMap = newDimensions.enumerated().filter{ $0.element == 1 }.map{ $0.offset }
        var broadcastedData = data
        
        for i in (0..<newDimensions.count).reversed() {
            if newDimensions[i] == 1 && targetDimensions[i] > 1 {
                if i == newDimensions.count - 1 {
                    let returnCount = targetDimensions[i] - newDimensions[i]
                    let count = dataSize / newDimensions.last!
                    /*
                    for row in 0..<count {
                        broadcastedData.insert(contentsOf: Array(repeating: data[row], count: returnCount), at: row * targetDimensions[i])
                    }*/
                    var a = 0
                    for i in 0..<1000 {
                        a += 3
                    }
                } else {
                    print("DIAGNOSTIC STATISTICS")
                    let t1 = CFAbsoluteTimeGetCurrent()
                    let returnCount = targetDimensions[i] - newDimensions[i]
                    let t2 = CFAbsoluteTimeGetCurrent()
                    let t3 = CFAbsoluteTimeGetCurrent()
                    //broadcastedData.append(contentsOf: repeatArray(broadcastedData, count: returnCount))
                    //Start RepeatArray
                    
                    
                    // Calculate the total length of the resulting array
                    let repeatedLength = broadcastedData.count * returnCount
                    if T.self == Float.self {
                        let ttt1 = CFAbsoluteTimeGetCurrent()
                        // Create an output array with the required length, initialized to zero
                        var result = [Float](repeating: 0, count: repeatedLength)
                        
                        // Use unsafe mutable buffer pointers for efficient copying
                        result.withUnsafeMutableBufferPointer { resultPointer in
                            // Get a pointer to the start of the result buffer
                            //let resultBase = resultPointer.baseAddress!
                            guard let resultBase = resultPointer.baseAddress else {
                                fatalError("Result base address is nil")
                            }
                            // Copy the original array into the result buffer for the first time
                            broadcastedData.withUnsafeBufferPointer { arrayPointer in
                                // Get a pointer to the start of the array buffer
                                guard let arrayBase = arrayPointer.baseAddress else {
                                    fatalError("Array base address is nil")
                                }
                                // Copy the original array into the result buffer
                                vDSP_mmov(arrayBase as! UnsafePointer<Float>, resultBase, vDSP_Length(broadcastedData.count), 1, vDSP_Length(broadcastedData.count), 1)
                                // Repeat the copy operation using exponential growth
                                var currentLength = broadcastedData.count
                                while currentLength < repeatedLength {
                                    // Double the size of copied elements each time
                                    let remainingLength = min(currentLength, repeatedLength - currentLength)
                                    vDSP_mmov(resultBase, resultBase + currentLength, vDSP_Length(remainingLength), 1, vDSP_Length(remainingLength), 1)
                                    currentLength += remainingLength
                                }
                            }
                        }
                        let ttt2 = CFAbsoluteTimeGetCurrent()
                        print("Inner repeatArray: \(ttt2 - ttt1)")
                        broadcastedData.append(contentsOf: result as! [T])
                    }


                    // End RepeatArray
                    let t4 = CFAbsoluteTimeGetCurrent()
                    //broadcastedData.append(contentsOf: array)
                    let t5 = CFAbsoluteTimeGetCurrent()
                    newDimensions[i] = targetDimensions[i]
                    
                    print("Calculated internal properties: \(t2 - t1)")
                    print("Generated array: \(t3 - t2)")
                    print("Generated array data: \(t4 - t3)")
                    print("Applied data: \(t5 - t4)")
                    print("Total time: \(t5 - t1)")
                }
            }
        }
        
        return self
    }
    
    @inlinable
    public func indexToFlatIndex(_ indices: [Int]) -> Int {
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
    public func backward(_ grad: [T] = Array(repeating: 1.0, count: 0), printSteps: Bool = false) {
        // Check if the gradient sizes match
        let grad = grad.isEmpty ? [T](repeating: 1.0, count: gradient!.count) : grad
        // Accumulate gradients
        for i in 0..<gradient!.count {
            gradient![i] += grad[i]
        }
        //gradient = add(gradient, grad)
        
        if printSteps {
            print("Grad At Operation: \(operation ?? "Leaf")")
            print(gradient!)
        }
        
        for (parent, local) in parents {
            guard parent.gradient != nil else {
                continue
            }
            //if parent.calculate_grad {
                let localGradients = local(self)
                //let combinedGradients = localGradients
                parent.backward(localGradients, printSteps: printSteps)
            //}
        }
    }
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
public func calculateIndex(strides: [Int], index: [Int]) -> Int {
    let a = zip(strides, index).map(*)
    return a.reduce(0, +)
}
@inlinable
public func sum<T: TensorType>(_ input: [T], shape: [Int], along: Int) -> [T] {
    // Ensure the dimension is valid
    precondition(along < shape.count, "Invalid dimension for sum")

    let dimSize = shape[along]
    var summedValues = [T](repeating: 0.0, count: input.count / dimSize)

    // Calculate the stride and number of blocks for the specified dimension
    let numBlocks = input.count / dimSize
    let blockStride = shape.dropFirst(along + 1).reduce(1, *)

    // Perform the summation along the specified dimension
    for blockIndex in 0..<numBlocks {
        // Compute the starting index for this block
        let blockStartIndex = (blockIndex / blockStride) * (blockStride * dimSize) + (blockIndex % blockStride)

        for j in 0..<dimSize {
            let index = blockStartIndex + j * blockStride
            summedValues[blockIndex] += input[index]
        }
    }

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
public func sum<T: TensorType>(_ data: [T], shape: [Int], along: [Int]) -> [T] {
    var result = data
    for i in along {
        result = sum(result, shape: shape, along: i)
    }
    return result
}

@inlinable
public func sum<T: TensorType>(_ data: [T], shape: [Int], to: [Int]) -> [T] {
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
public func sum<T: TensorType>(_ data: [T], shape: [Int]) -> T {
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
public func multiply<T: TensorType>(_ input: [T], s: T) -> [T] {
    var result = [T]()
    let totalElements = input.count
    
    if T.self == Float.self {
        var outputData = [Float](repeating: 0, count: totalElements)
        
        input.withUnsafeBufferPointer { rBuffer in
            [Float](repeating: s as! Float, count: rBuffer.count).withUnsafeBufferPointer{ sd in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vsmul(
                            rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                            sd.baseAddress!,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(totalElements)
                        )
                }
            }
        }
        
        result = outputData as! [T]
    } else if T.self == Double.self {
        var outputData = [Double](repeating: 0, count: totalElements)
        
        input.withUnsafeBufferPointer { rBuffer in
            [Double](repeating: s as! Double, count: rBuffer.count).withUnsafeBufferPointer{ sd in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vsmulD(
                            rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                            sd.baseAddress!,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(totalElements)
                        )
                }
            }
        }
        
        result = outputData as! [T]
    } else /*if T.self == Float16.self*/ {
        var outputData = [Float](repeating: 0, count: totalElements)
        
        let lDataFloat = input.compactMap { Float($0) }
        lDataFloat.withUnsafeBufferPointer { rBuffer in
            [Float](repeating: Float(s), count: rBuffer.count).withUnsafeBufferPointer{ sd in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vsmul(
                            rBuffer.baseAddress!, 1,
                            sd.baseAddress!,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(totalElements)
                        )
                }
            }
        }
        result = outputData.compactMap{ T($0) }
    }
    return result
}

@inlinable
public func divide<T: TensorType>(_ input: [T], s: T) -> [T] {
    var result = [T]()
    let totalElements = input.count
    
    if T.self == Float.self {
        var outputData = [Float](repeating: 0, count: totalElements)
        
        input.withUnsafeBufferPointer { rBuffer in
            [Float](repeating: s as! Float, count: rBuffer.count).withUnsafeBufferPointer{ sd in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vdiv(
                        sd.baseAddress!, 1,
                        rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        oBuffer.baseAddress!, 1,
                        vDSP_Length(totalElements)
                    )
                }
            }
        }
        
        result = outputData as! [T]
    } else if T.self == Double.self {
        var outputData = [Double](repeating: 0, count: totalElements)
        
        input.withUnsafeBufferPointer { rBuffer in
            [Double](repeating: s as! Double, count: rBuffer.count).withUnsafeBufferPointer{ sd in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vdivD(
                            sd.baseAddress!, 1,
                            rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(totalElements)
                        )
                }
            }
        }
        
        result = outputData as! [T]
    } else /*if T.self == Float16.self*/ {
        var outputData = [Float](repeating: 0, count: totalElements)
        
        let lDataFloat = input.compactMap { Float($0) }
        lDataFloat.withUnsafeBufferPointer { rBuffer in
            [Float](repeating: Float(s), count: rBuffer.count).withUnsafeBufferPointer{ sd in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vdiv(
                        sd.baseAddress!, 1,
                        rBuffer.baseAddress!, 1,
                        oBuffer.baseAddress!, 1,
                        vDSP_Length(totalElements)
                    )
                }
            }
        }
    }
    return result
}

@inlinable
public func inverseDivide<T: TensorType>(_ input: [T], s: T) -> [T] {
    var result = [T]()
    let totalElements = input.count
    
    if T.self == Float.self {
        var outputData = [Float](repeating: 0, count: totalElements)
        
        input.withUnsafeBufferPointer { rBuffer in
            [Float](repeating: s as! Float, count: rBuffer.count).withUnsafeBufferPointer{ sd in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vdiv(
                        rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                        sd.baseAddress!, 1,
                        oBuffer.baseAddress!, 1,
                        vDSP_Length(totalElements)
                    )
                }
            }
        }
        
        result = outputData as! [T]
    } else if T.self == Double.self {
        var outputData = [Double](repeating: 0, count: totalElements)
        
        input.withUnsafeBufferPointer { rBuffer in
            [Double](repeating: s as! Double, count: rBuffer.count).withUnsafeBufferPointer{ sd in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                        vDSP_vdivD(
                            rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                            sd.baseAddress!, 1,
                            oBuffer.baseAddress!, 1,
                            vDSP_Length(totalElements)
                        )
                }
            }
        }
        
        result = outputData as! [T]
    } else /*if T.self == Float16.self*/ {
        var outputData = [Float](repeating: 0, count: totalElements)
        
        let lDataFloat = input.compactMap { Float($0) }
        lDataFloat.withUnsafeBufferPointer { rBuffer in
            [Float](repeating: Float(s), count: rBuffer.count).withUnsafeBufferPointer{ sd in
                outputData.withUnsafeMutableBufferPointer { oBuffer in
                    vDSP_vdiv(
                        rBuffer.baseAddress!, 1,
                        sd.baseAddress!, 1,
                        oBuffer.baseAddress!, 1,
                        vDSP_Length(totalElements)
                    )
                }
            }
        }
    }
    return result
}

@inlinable
public func matrixMultiply<T: TensorType>(_ a: [T], _ b: [T], aShape: [Int], bShape: [Int]) -> [T] {
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
            var outputData = [Float](repeating: 0, count: outputSize)
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
                                oBuffer.baseAddress! + oOffset, 1,
                                vDSP_Length(aRows),
                                vDSP_Length(bCols),
                                vDSP_Length(aCols)
                            )
                        }
                    }
                }
            }
            result = outputData as! [T]
        } else if T.self == Double.self {
            var outputData = [Double](repeating: 0, count: outputSize)
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
                                oBuffer.baseAddress! + oOffset, 1,
                                vDSP_Length(aRows),
                                vDSP_Length(bCols),
                                vDSP_Length(aCols)
                            )
                        }
                    }
                }
            }
            result = outputData as! [T]
        } else /*if T.self == Float16.self*/ {
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
                            vDSP_mmul(
                                lBuffer.baseAddress! + lOffset, 1,
                                rBuffer.baseAddress! + rOffset, 1,
                                oBuffer.baseAddress! + oOffset, 1,
                                vDSP_Length(aRows),
                                vDSP_Length(bCols),
                                vDSP_Length(aCols)
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
public func transpose<T: TensorType>(_ input: [T], shape: [Int]) -> [T] {
    var result = [T]()
    let rows = shape.dropLast().last!
    let cols = shape.last!
    let totalElements = input.count
    let batchCount = shape.dropLast(2).reduce(1, *)
    
    if T.self == Float.self {
        var outputData = [Float](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            input.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtrans(sdata.baseAddress! as! UnsafePointer<Float> + offset, 1, odata.baseAddress! + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData as! [T]
    } else if T.self == Double.self {
        var outputData = [Double](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            input.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtransD(sdata.baseAddress! as! UnsafePointer<Double> + offset, 1, odata.baseAddress! + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData as! [T]
    } else /*if T.self == Float16.self*/ {
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
public func transpose<T: TensorType>(_ input: Tensor<T>) -> Tensor<T> {
    var result = [T]()
    let rows = input.shape.dropLast().last!
    let cols = input.shape.last!
    let totalElements = input.data.count
    let batchCount = input.shape.dropLast(2).reduce(1, *)
    
    if T.self == Float.self {
        var outputData = [Float](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            input.data.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtrans(sdata.baseAddress! as! UnsafePointer<Float> + offset, 1, odata.baseAddress! + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData as! [T]
    } else if T.self == Double.self {
        var outputData = [Double](repeating: 0, count: totalElements)
        for batch in 0..<batchCount {
            let offset = batch * rows * cols
            input.data.withUnsafeBufferPointer { sdata in
                outputData.withUnsafeMutableBufferPointer { odata in
                    vDSP_mtransD(sdata.baseAddress! as! UnsafePointer<Double> + offset, 1, odata.baseAddress! + offset, 1, vDSP_Length(cols), vDSP_Length(rows))
                }
            }
        }
        result = outputData as! [T]
    } else /*if T.self == Float16.self*/ {
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
