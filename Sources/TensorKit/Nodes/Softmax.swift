//
//  Softmax.swift
//  Synapse
//
//
/*
* Copyright (c) 2024 Nectralyx.
* This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*/

import Foundation

internal func softmaxJacobian<T: TensorType>(_ y: Tensor<T>) -> [T] {
    
    let dimension = y.shape.count - 1
    let strides = generateStrides(y.shape)
    var jacobianData = [T](unsafeUninitializedCapacity: y.dataSize * y.shape[dimension]) { buffer, initializedCount in
        // Manually set values in the buffer

        initializedCount = buffer.count  // Set to the number of elements you actually initialize
    }
    
    // Calculate outer and inner dimensions based on the specified dimension
    
    let adjusted = dimension
    let dimSize = y.shape[adjusted]
    
    // Calculate total number of iterations across dimensions
    let totalIterations = y.shape.reduce(1, *)
    let jSize = y.shape.last! * y.shape.last!
    var interoutputindex = 0
    for iteration in 0..<totalIterations {
        
        // Determine current position in the tensor
        var index = iteration
        var currentIndices = [Int](repeating: 0, count: y.shape.count)
        
        // Decode multi-dimensional index from linear index
        for i in stride(from: y.shape.count - 1, through: 0, by: -1) {
            currentIndices[i] = index % y.shape[i]
            index /= y.shape[i]
        }
        
        // Only act when the dimension to slice matches the current index
        if currentIndices[adjusted] == 0 {
            
            var slice = [T]()
            
            // Collect the elements for the slice across the desired dimension
            for sliceIndex in 0..<dimSize {
                currentIndices[adjusted] = sliceIndex
                let flatIndex = zip(currentIndices, strides).map(*).reduce(0, +)
                slice.append(y.data[flatIndex])
            }
            
            //ACTION v
            let sliceSize = slice.count
            var jMatrix = [T](repeating: 0, count: sliceSize * sliceSize)
            for i in 0..<sliceSize {
                for j in 0..<sliceSize {
                    jMatrix[i * sliceSize + j] = i == j ? slice[i] * (1 - slice[i]) : -(slice[i] * slice[j])
                }
            }
            let jIndex = interoutputindex * jSize
            let jEdge = (interoutputindex + 1) * jSize
            jacobianData[jIndex..<jEdge] = ArraySlice(jMatrix)
            //ACTION ^
            interoutputindex += 1
        }
    }
   
   
    return jacobianData
}

internal func softmaxGradients<T: TensorType>(_ jacobians: [T], jShape: [Int], outputGrads: [T], outputShape: [Int]) -> [T] {
    let jCount = outputShape.reduce(1, *) / outputShape.last!
    let strides = generateStrides(jShape)
    _ = generateStrides(outputShape)
    var result = [T]()
    _ = DispatchQueue(label: "x10.mx.Synapse.Softmax.softmaxGradients")
    
    for i in 0..<jCount {
        let index = calculateIndex(strides: strides, index: Array(repeating: 0, count: outputShape.count).inserting(i, at: jShape.count - 3))
        let nextEdge = calculateIndex(strides: strides, index: Array(repeating: 0, count: outputShape.count).inserting(i + 1, at: jShape.count - 3))
        let gradIndex = calculateIndex(strides: strides, index: Array(repeating: 0, count: outputShape.count - 1).inserting(i, at: outputShape.count - 1))
        let gradEdge = calculateIndex(strides: strides, index: Array(repeating: 0, count: outputShape.count - 1).inserting(i + 1, at: outputShape.count - 1))
        let js = Array(jacobians[index..<nextEdge])
        let grad = Array(outputGrads[gradIndex..<gradEdge])
        let value = matrixMultiply(js, grad, aShape: [jShape.last!, jShape.last!], bShape: [outputShape.last!, 1])
        result.append(contentsOf: value)
    }
    return result
}

public func Softmax<T: TensorType>(_ input: Tensor<T>) -> Tensor<T> {
    // Ensure the dimension is valid
    let dimension = input.shape.count - 1
    precondition(dimension < input.shape.count, "Invalid dimension for softmax")
    
    let shape = input.shape
    let dimSize = shape[dimension]
    var softmaxValues = [T](repeating: 0.0, count: input.data.count)
    
    // Calculate the stride and number of blocks for the specified dimension
    let numBlocks = input.data.count / dimSize
    let blockStride = shape.dropFirst(dimension + 1).reduce(1, *)
    
    // Compute exponentiated values using the log-exp-sum trick
    for blockIndex in 0..<numBlocks {
        // Compute the starting index for this block
        let blockStartIndex = (blockIndex / blockStride) * (blockStride * dimSize) + (blockIndex % blockStride)
        
        // Find the max value in the current block
        var maxVal = T.leastNonzeroMagnitude
        for j in 0..<dimSize {
            let index = blockStartIndex + j * blockStride
            maxVal = max(maxVal, input.data[index])
        }
        
        // Calculate the exponentiated values and sum for the block
        var expSum: T = 0.0
        for j in 0..<dimSize {
            let index = blockStartIndex + j * blockStride
            let shiftedValue = input.data[index] - maxVal
            softmaxValues[index] = T(exp(Double(shiftedValue)))
            expSum += softmaxValues[index]
        }
        
        // Normalize the exponentiated values by the sum
        for j in 0..<dimSize {
            let index = blockStartIndex + j * blockStride
            softmaxValues[index] /= expSum
        }
    }
    
    let result = Tensor(softmaxValues, shape: input.shape, calculate_grad: input.gradient != nil)
    let jShape = result.shape.inserting(result.shape.last!, at: result.shape.count - 1)
    result.operation = "Softmax"
    result.parents = [
        (input, { v in
            softmaxGradients(softmaxJacobian(result), jShape: jShape, outputGrads: v.gradient!, outputShape: v.shape)
        })
    ]
    return result
}
