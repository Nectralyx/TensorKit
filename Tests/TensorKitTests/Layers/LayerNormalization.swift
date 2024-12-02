//
//  LayerNormalization.swift
//  Synapse
//
//  Created by Morgan Keay on 2024-08-14.
//

import Foundation
import TensorKit

class LayerNormalization<T: TensorComplex>: Layer {
    var weight: Parameter<T>
    var bias: Parameter<T>
    var parameters: [Parameter<T>]
    var epsilon: Tensor<T> = Tensor(1e-5)
    var normDimensions: [Int]
    private var numDimensions: Int
    
    init(_ normalize_to: [Int], epsilon: Tensor<T> = Tensor(1e-5)) {
        self.epsilon = epsilon
        self.weight = Parameter(.ones, shape: normalize_to)
        self.bias = Parameter(.zeros, shape: normalize_to)
        self.parameters = [weight, bias]
        self.normDimensions = normalize_to
        self.numDimensions = normalize_to.count
        weight.operation = "Layer Normalization Weight"
        bias.operation = "Layer Normalization Bias"
    }
    
    func forward(_ input: Tensor<T>) -> Tensor<T> {
        var newCalc: [Int] = []
        let inputCount = input.shape.count
        
        for i in (1...numDimensions).reversed() {
            newCalc.append(inputCount - i)
        }

        let mean = input.mean(newCalc)
        let variance = input.variance(newCalc)
        let top = input - mean
        let newEPS = epsilon.expand(to: variance.shape)
        let bottom = sqrt(variance + newEPS)
        let x_norm = top / bottom
        let output = x_norm * weight + bias
        x_norm.disown()
        output.parents.append(
            (input, { v in
                var HShape = 1
                for i in 0..<newCalc.count {
                    HShape *= input.shape[input.shape.count - 1 - i]
                }
                
                bottom.operation = "Bottom"
                let rvar = 1.0 / bottom
                let rH = 1.0 / T(HShape)
                let dldy = Tensor(v.gradient!, shape: v.shape)
                let a = dldy * self.weight / bottom
                let b = rH * (dldy * self.weight * rvar).sum(along: newCalc)
                let c = rH * top * (dldy * self.weight * top * pow(bottom, -3)).sum(along: newCalc)
                
                let dldx = (a - b - c)
                return dldx.data
            })
        )
        return output
    }
}

