//
//  Embedding.swift
//  Synapse
//
//  Created by Morgan Keay on 2024-08-11.
//

import Foundation
import Accelerate
import TensorKit

class Embedding<T: TensorComplex>: Codable {
    var parameters: [Parameter<T>]
    var weights: Parameter<T>
    init(embeddings: Int, dimensions: Int, initializer: TensorInitialization = .random_small) {
        self.weights = Parameter(initializer, shape: [embeddings, dimensions])
        weights.operation = "Embedding Weight"
        self.parameters = [weights]
    }
    
    func forward(_ input: Tensor<T>) -> Tensor<T> {
        var resData = [T](unsafeUninitializedCapacity: input.dataSize * weights.shape[1], initializingWith: {  buffer, initializedCount in
            initializedCount = buffer.count
        })
        let staticSize = weights.shape[1]
        for i in 0..<input.dataSize {
            let start = Int(input.data[i]) * staticSize
            let end = (Int(input.data[i]) + 1) * staticSize
            resData[i * staticSize..<(i + 1) * staticSize] = weights.data[start..<end]
        }
        let result = Tensor(resData, shape: input.shape + [staticSize], calculate_grad: true)
        result.operation = "Embeddings"
        result.parents = [
            (weights, { v in
                var grads = [T](unsafeUninitializedCapacity: self.weights.dataSize, initializingWith: {  buffer, initializedCount in
                    initializedCount = buffer.count
                })
                let gradStride = stride(from: 0, to: staticSize, by: 1)
                for i in 0..<input.dataSize {
                    let index = Int(input.data[i])
                    let weightStart = index * staticSize
                    let gradStart = i * staticSize
                    
                    for j in gradStride {
                        grads[weightStart + j] += v.gradient![gradStart + j]
                    }
                }
                return grads
            }
            )
        ]
        return result
    }
}
