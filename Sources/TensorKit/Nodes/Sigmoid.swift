//
//  Sigmoid.swift
//  Synapse
//
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import Foundation

@inlinable
public func Sigmoid<T: TensorType>(_ input: Tensor<T>) -> Tensor<T> {
    let result = Tensor<T>(.empty, shape: input.shape, calculate_grad: input.gradient != nil)
    result.data = input.data.map{ Sigmoid($0) }
    result.operation = "Sigmoid"
    result.parents = [
        (input, { v in v.gradient!.map{ SigmoidDerivative($0) * $0 } })
    ]
    return result
}
