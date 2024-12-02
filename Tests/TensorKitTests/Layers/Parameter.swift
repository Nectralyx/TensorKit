//
//  Parameter.swift
//  Synapse
//
// 
//

import Foundation
import TensorKit

class Parameter<T: TensorComplex>: Tensor<T> {
    var parameterCount: Int {
        return data.count
    }
    
    override init(_ input: Tensor<T>) {
        super.init(input.data, shape: input.shape)
        self.shape = input.shape
        self.gradient = input.gradient != nil ? input.gradient : [T](repeating: 0.0, count: data.count)
        self.data = input.data
    }
    
    override init(_ input: [[[T]]], calculate_grad: Bool = true) {
        super.init(input)
        self.shape = [input.count, input[0].count, input[0][0].count]
        self.data = input.flatMap { $0.flatMap { $0 } }
        self.gradient = [T](repeating: 0.0, count: data.count)
    }
    
    override init(_ input: [[T]], calculate_grad: Bool = true) {
        super.init(input)
        self.shape = [input.count, input[0].count]
        self.data = input.flatMap{ $0 }
        self.gradient = [T](repeating: 0.0, count: data.count)
    }
    
    override init(_ input: [T], shape: [Int], calculate_grad: Bool = true) {
        super.init(input, shape: shape)
        self.shape = shape
        self.data = input
        self.gradient = [T](repeating: 0.0, count: data.count)
    }
    override init(_ input: T, calculate_grad: Bool = true) {
        super.init(input)
        self.shape = [1]
        self.data = [input]
        self.gradient = [0.0]
    }
    
    override init(_ initializer: TensorInitialization = .zeros, shape: [Int], calculate_grad: Bool = true) {
        super.init(shape: shape)
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
            let xavierScale = sqrt(6.0 / Double(shape[shape.endIndex - 1] + shape[shape.endIndex - 2]))
            self.data = [T](repeating: T(Double.random(in: -xavierScale...xavierScale)), count: shape.reduce(1, *))
        case .he:
            let heScale = sqrt(6.0 / Double(shape[shape.endIndex - 2]))
            self.data = [T](repeating: T(Double.random(in: -heScale...heScale)), count: shape.reduce(1, *))
        case .random:
            self.data = [T](repeating: T(Double.random(in: -10000...10000)), count: shape.reduce(1, *))
        case .empty:
            self.data = []
        case .random_small:
            self.data = []
            for _ in 0..<shape.reduce(1, *) {
                self.data.append(T(Double.random(in: -0.01...0.01)))
            }
        }
        self.gradient = [T](repeating: 0.0, count: data.count)
    }
    
    override func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(data, forKey: .data)
        try container.encode(shape, forKey: .shape)
        try container.encode(operation, forKey: .operation)
        try container.encode(gradient, forKey: .gradient)
    }
    
    required init(from decoder: any Decoder) throws {
        try super.init(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.data = try container.decode([T].self, forKey: .data)
        self.shape = try container.decode([Int].self, forKey: .shape)
        self.operation = try container.decode(String?.self, forKey: .operation)
        self.gradient = try container.decode([T].self, forKey: .gradient)
    }
    
    enum CodingKeys: CodingKey {
        case data
        case shape
        case operation
        case gradient
    }
    
}
