//
//  Linear.swift
//  Synapse
//
//  Created by Morgan Keay on 2024-08-14.
//

import Foundation
import TensorKit

class Linear<T: TensorComplex>: Layer {
    var weights: Parameter<T>
    var biases: Parameter<T>
    var parameters: [Parameter<T>]
    init(inputSize: Int, outputSize: Int, initializer: TensorInitialization = .xavier_glorot) {
        self.weights = Parameter<T>(initializer, shape: [outputSize, inputSize])
        self.biases = Parameter<T>(initializer, shape: [1, outputSize])
        self.parameters = [weights, biases]
        weights.operation = "Linear Weight"
        biases.operation = "Linear Bias"
    }
    
    func forward(_ input: Tensor<T>) -> Tensor<T> {
        let a = input ** weights.transpose()
        let b = a + biases
        return b
    }
}
