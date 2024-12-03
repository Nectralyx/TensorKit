//
//  conformances.swift
//  Nucleus
//
//  
//

import Foundation
import TensorKit

protocol Layer<T>: Codable {
    associatedtype T: TensorComplex
    var parameters: [Parameter<T>] { get }
    func forward(_ input: Tensor<T>) -> Tensor<T>
}

protocol Optimizer: Codable {
}

enum TokenBase: String, Codable, CaseIterable, Identifiable {
    var id: Self {
        return self
    }
    case basic
    case addNumerics
    case addSymbols
    case addEmojis
    case numericsAndSymbols
    case numericsAndEmojis
    case symbolsAndEmojis
    case all
}

protocol Model: Codable {
}

protocol LossFunction: Codable {
}
