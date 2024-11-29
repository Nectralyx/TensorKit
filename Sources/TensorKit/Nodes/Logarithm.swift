//
//  Logarithm.swift
//  TensorKit
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import Foundation
import TensorKit
import Accelerate

@inlinable
public func log<T: TensorComplex>(_ x: Tensor<T>) -> Tensor<T> {
    let result = Tensor<T>(.zeros, shape: x.shape, calculate_grad: x.gradient != nil)
    let count = Int32(x.dataSize)
    if T.self == Float.self {
        result.data.withUnsafeMutableBufferPointer{ rBuffer in
            x.data.withUnsafeBufferPointer{ xBuffer in
                [count].withUnsafeBufferPointer{ cBuffer in
                    vvlogf(
                        rBuffer.baseAddress! as! UnsafeMutablePointer<Float>,
                        xBuffer.baseAddress! as! UnsafePointer<Float>,
                        cBuffer.baseAddress!
                    )
                }
            }
        }
    } else if T.self == Double.self {
        result.data.withUnsafeMutableBufferPointer{ rBuffer in
            x.data.withUnsafeBufferPointer{ xBuffer in
                [count].withUnsafeBufferPointer{ cBuffer in
                    vvlog(
                        rBuffer.baseAddress! as! UnsafeMutablePointer<Double>,
                        xBuffer.baseAddress! as! UnsafePointer<Double>,
                        cBuffer.baseAddress!
                    )
                }
            }
        }
    }
    result.operation = "Natural Logarithm"
    result.parents = [
        (x, { v in
            return inverseDivide(x.data, s: 1.0)
        })
    ]
    return result
}
