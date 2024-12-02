//
//  MultiHeadAttention.swift
//  Synapse
//
//  Created by Morgan Keay on 2024-05-29.
//

import Foundation
import TensorKit

class multiHeadAttention<T: TensorComplex>: Layer {
    let keyProjector: Linear<T>
    let queryProjector: Linear<T>
    let valueProjector: Linear<T>
    let outputProjector: Linear<T>
    let num_heads: Int
    let model_size: Int
    var mask: Bool
    var head_size: Int {
        model_size / num_heads
    }
    var parameters: [Parameter<T>]
    init(model_size: Int, num_heads: Int, mask: Bool = false, initializer: TensorInitialization = .xavier_glorot) {
        guard model_size % num_heads == 0 else {
            fatalError("Number of heads must divide model size")
        }
        
        keyProjector = Linear(inputSize: model_size, outputSize: model_size, initializer: initializer)
        queryProjector = Linear(inputSize: model_size, outputSize: model_size, initializer: initializer)
        valueProjector = Linear(inputSize: model_size, outputSize: model_size, initializer: initializer)
        outputProjector = Linear(inputSize: model_size, outputSize: model_size, initializer: initializer)
        
        self.model_size = model_size
        self.num_heads = num_heads
        self.mask = mask
        parameters = [keyProjector.parameters, queryProjector.parameters, valueProjector.parameters, outputProjector.parameters].flatMap{ $0 }
        keyProjector.weights.operation = "Multihead Attention Key Projector Weight"
        keyProjector.biases.operation = "Multihead Attention Key Projector Bias"
        queryProjector.weights.operation = "Multihead Attention Query Projector Weight"
        queryProjector.biases.operation = "Multihead Attention Query Projector Bias"
        valueProjector.weights.operation = "Multihead Attention Value Projector Weight"
        valueProjector.biases.operation = "Multihead Attention Value Projector Bias"
        outputProjector.weights.operation = "Multihead Attention Output Projector Weight"
        outputProjector.biases.operation = "Multihead Attention Output Projector Bias"
    }
    
    func forward(_ input: Tensor<T>) -> Tensor<T> {
        guard model_size == input.shape.last! else {
            fatalError("Incorrect model size \(input.shape.last!), Expected \(model_size)")
        }
        let originalShape = input.shape[0..<input.shape.count - 2]
        let batch_size = originalShape.reduce(1, *)
        let seq_length = input.shape.dropLast().last!
        
        var K = keyProjector.forward(input)
        var Q = queryProjector.forward(input)
        var V = valueProjector.forward(input)

        if K.shape.count > 2 {
            K = K.view([batch_size, seq_length, num_heads, head_size])
            Q = Q.view([batch_size, seq_length, num_heads, head_size])
            V = V.view([batch_size, seq_length, num_heads, head_size])
            
            K = K.permute([0, 2, 1, 3])
            Q = Q.permute([0, 2, 1, 3])
            V = V.permute([0, 2, 1, 3])
        } else if K.shape.count <= 2 {
            K = K.view([seq_length, num_heads, head_size])
            Q = Q.view([seq_length, num_heads, head_size])
            V = V.view([seq_length, num_heads, head_size])
            
            K = K.permute([1, 0, 2])
            Q = Q.permute([1, 0, 2])
            V = V.permute([1, 0, 2])
        }
        var result = SDPA(K, Q, V, mask)
        result.operation = "SDPA Weight"
        if K.shape.count > 2 {
            result = result.permute([0, 2, 1, 3])
        } else if K.shape.count <= 2 {
            result = result.permute([1, 0, 2])
        }
        
        result = result.view(originalShape + [seq_length] + [model_size])
        let output = outputProjector.forward(result)
        return output
    }
    
    func forward(_ keys: Tensor<T>, _ queries: Tensor<T>, _ values: Tensor<T>, _ ignore_mask: Tensor<T>? = nil) -> Tensor<T> {
        guard model_size == keys.shape.last! else {
            fatalError("Incorrect model size \(keys.shape.last!), Expected \(model_size)")
        }
        let originalShape = keys.shape[0..<keys.shape.count - 2]
        let batch_size = originalShape.reduce(1, *)
        let seq_length = keys.shape.dropLast().last!
        
        var K = keyProjector.forward(keys)
        var Q = queryProjector.forward(queries)
        var V = valueProjector.forward(values)
        if K.shape.count > 2 {
            K = K.view([batch_size, seq_length, num_heads, head_size])
            Q = Q.view([batch_size, seq_length, num_heads, head_size])
            V = V.view([batch_size, seq_length, num_heads, head_size])
            K = K.permute([0, 2, 1, 3])
            Q = Q.permute([0, 2, 1, 3])
            V = V.permute([0, 2, 1, 3])
        } else if K.shape.count <= 2 {
            K = K.view([seq_length, num_heads, head_size])
            Q = Q.view([seq_length, num_heads, head_size])
            V = V.view([seq_length, num_heads, head_size])
            K = K.permute([1, 0, 2])
            Q = Q.permute([1, 0, 2])
            V = V.permute([1, 0, 2])
        }
        var result = SDPA(K, Q, V, mask, ignore_mask)
       
        if K.shape.count > 2 {
            result = result.permute([0, 2, 1, 3])
        } else if K.shape.count <= 2 {
            result = result.permute([1, 0, 2])
        }
        result = result.view(originalShape + [seq_length] + [model_size])

        let output = outputProjector.forward(result)
        return output
    }
}
