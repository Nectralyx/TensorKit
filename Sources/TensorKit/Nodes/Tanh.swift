//
//  Tanh.swift
//  TensorKit
//
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import Foundation

@inlinable
public func Tanh<T: TensorComplex>(_ input: Tensor<T>) -> Tensor<T> {
    let result = Tensor<T>(.empty, shape: input.shape, calculate_grad: input.gradient != nil)
    result.data = input.data.map{ Tanh($0) }
    result.operation = "Tanh"
    result.parents = [
        (input, { v in
            let map = input.map{ TanhDerivative($0) }
            return multiply(v.gradient!, map)
        })
    ]
    return result
}
