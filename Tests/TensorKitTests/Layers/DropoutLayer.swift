//
//  DropoutLayer.swift
//  Synapse
//
//  
//

import Foundation
import TensorKit
/*
class DropoutLayer<T: BinaryFloatingPoint & Codable>: oLayer, Codable {
    let probablility: T
    let parameters: Int = 0
    var mask: [[T]] = []
    init(probablility: T) {
        self.probablility = probablility
    }
    func forward(_ input: [[T]]) -> [[T]] {
        var maskedInput = input
        var masking: [[T]] = []
        for i in 0..<input.count {
            var row: [T] = []
            for j in 0..<input[0].count {
                let rand = T(Double.random(in: 0...1))
                row.append(rand > probablility ? T(1.0) : T(0.0))
                maskedInput[i][j] *= rand > probablility ? T(1.0) : T(0.0)
            }
            masking.append(row)
        }
        mask = masking
        /*let masked = input.map{
            $0.map{
                let rand = Double.random(in: 0..<1)
                return probablility > rand ? $0 : 0
            }
        } */
        return maskedInput.map{ $0.map{ $0 / (1 - probablility + 1e-5) } }
    }
    
    func backward(input: [[T]], passthrough: [[T]], learningRate: T) -> [[T]] {
       /* print("Test")
        for i in passthrough {
            print(i)
        }
        let ret = zip(passthrough, mask).map{ zip($0.0, $0.1).map{ $0.0 * $0.1 / (1 - probablility)}}
        for i in ret {
            print(i)
        } */
        return zip(passthrough, mask).map{ zip($0.0, $0.1).map{ $0.0 * $0.1 / (1 - probablility + 1e-5)}}
    }
    
    
}
*/

func dropOut<T: TensorComplex>(_ input: Tensor<T>, prob: T) -> Tensor<T> {
    let mask = Tensor<T>(.ones, shape: input.shape, calculate_grad: input.gradient != nil)
    for i in 0..<input.dataSize {
        let rand = T(Double.random(in: 0..<1))
        if rand < prob {
            mask.data[i] = 0
        }
    }
    
    let output = mask * input
    output.operation = "Dropout"
    output.parents = [
        (input, { v in multiply(v.gradient!, mask.data)})
    ]
    return output
}
