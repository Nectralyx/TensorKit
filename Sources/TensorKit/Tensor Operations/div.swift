//
//  div.swift
//  Synapse
//
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import Foundation
import Accelerate

public extension Tensor {
    static func /(lhs: Tensor, rhs: Tensor) -> Tensor {
        let finalShape = mergeShapes(lhs.shape, rhs.shape)
        let lhs = lhs.expand(to: finalShape)
        let rhs = rhs.expand(to: finalShape)
        let result = Tensor(.empty, shape: finalShape, calculate_grad: (lhs.gradient != nil || rhs.gradient != nil) ? true : false)
            if T.self == Float.self {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { rBuffer in
                    rhs.data.withUnsafeBufferPointer { lBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                                vDSP_vdiv(
                                    lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                    rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                    oBuffer.baseAddress!, 1,
                                    vDSP_Length(result.dataSize)
                                )
                        }
                    }
                }
                
                result.data = outputData as! [T]
            } else if T.self == Double.self {
                var outputData = [Double](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { rBuffer in
                    rhs.data.withUnsafeBufferPointer { lBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                                vDSP_vdivD(
                                    lBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                                    rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                                    oBuffer.baseAddress!, 1,
                                    vDSP_Length(result.dataSize)
                                )
                        }
                    }
                }
                
                result.data = outputData as! [T]
            } else /*if T.self == Float16.self*/ {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                
                let lDataFloat = lhs.data.compactMap { Float($0) }
                let rDataFloat = rhs.data.compactMap { Float($0) }
                
                lDataFloat.withUnsafeBufferPointer { rBuffer in
                    rDataFloat.withUnsafeBufferPointer { lBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                                vDSP_vdiv(
                                    lBuffer.baseAddress!, 1,
                                    rBuffer.baseAddress!, 1,
                                    oBuffer.baseAddress!, 1,
                                    vDSP_Length(result.dataSize)
                                )
                        }
                    }
                }
            }
        
        result.operation = "/"
        result.parents = [
            (lhs, { v in multiply(inverseDivide(rhs.data, s: 1), v.gradient!) }),
            (rhs, { v in multiply(lhs.data.enumerated().map{ index, value in -value / (rhs.data[index] * rhs.data[index])}, v.gradient!) })
        ]
        return result
    }
}
