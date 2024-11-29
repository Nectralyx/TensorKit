//
//  Nodes.swift
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

@inlinable
public func ReLU<T: TensorComplex>(_ input: Tensor<T>, leak: T = 0) -> Tensor<T> {
    let result = Tensor<T>(.empty, shape: input.shape, calculate_grad: input.gradient != nil)
    if leak == T(0) {
        result.data = input.data.map{ ReLU($0) }
        result.parents = [
            (input, { v in
                let map = input.data.map{ ReLUDerivative($0) }
                return multiply(v.gradient!, map)
            })
        ]
    } else {
        result.data = input.data.map{ LeakyReLU($0, alpha: leak) }
        result.parents = [
            (input, { v in
                let map = input.data.map{ LeakyReLUDerivative($0, alpha: leak) }
                return multiply(v.gradient!, map)
            })
        ]
    }
    result.operation = "ReLU"
    return result
}



