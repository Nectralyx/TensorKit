//
//  div.swift
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

public extension Tensor {
    @inlinable
    static func /(lhs: Tensor, rhs: Tensor) -> Tensor {
        let finalShape = mergeShapes(lhs.shape, rhs.shape)
        let lhs = lhs.expand(to: finalShape)
        let rhs = rhs.expand(to: finalShape)
        let result = Tensor(.empty, shape: finalShape, calculate_grad: (lhs.gradient != nil || rhs.gradient != nil) ? true : false)
            if T.self == Float.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { rBuffer in
                    rhs.data.withUnsafeBufferPointer { lBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vdiv(
                                lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData
            } else if T.self == Double.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { rBuffer in
                    rhs.data.withUnsafeBufferPointer { lBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vdivD(
                                lBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                                rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Double>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData
            } else {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                
                let lDataFloat = lhs.data.map { Float($0) }
                let rDataFloat = rhs.data.map { Float($0) }
                
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
            (lhs, { v in
                return multiply(inverseDivide(rhs.data, s: 1), v.gradient!) }),
            (rhs, { v in
                let out = Tensor(-1) * lhs / pow(rhs, 2)
                return multiply(out.data, v.gradient!)})
        ]
        return result
    }
    
    @inlinable
    static func /(lhs: T, rhs: Tensor) -> Tensor {
        
        let finalShape = rhs.shape
        let lhsData = [lhs]
        let result = Tensor(.empty, shape: finalShape, calculate_grad: rhs.gradient != nil)
            if T.self == Float.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                
                
                lhsData.withUnsafeBufferPointer { lBuffer in
                    rhs.data.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_svdiv(
                                lBuffer.baseAddress! as! UnsafePointer<Float>,
                                rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData
            } else if T.self == Double.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                
                lhsData.withUnsafeBufferPointer { lBuffer in
                    rhs.data.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_svdivD(
                                lBuffer.baseAddress! as! UnsafePointer<Double>,
                                rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Double>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData
            } else {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                
                let lDataFloat = lhsData.map { Float($0) }
                let rDataFloat = rhs.data.map { Float($0) }
                
                lDataFloat.withUnsafeBufferPointer { lBuffer in
                    rDataFloat.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_svdiv(
                                lBuffer.baseAddress!,
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
            (rhs, { v in
                let out = inverseDivide(pow(rhs, 2).data, s: -lhs)
                return multiply(out, v.gradient!)})
        ]
        return result
    }
    
    @inlinable
    static func /(lhs: Tensor, rhs: T) -> Tensor {
        let finalShape = lhs.shape
        let rhsData = [rhs]
        let result = Tensor(.empty, shape: finalShape, calculate_grad: lhs.gradient != nil)
            if T.self == Float.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { rBuffer in
                    rhsData.withUnsafeBufferPointer { lBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vsdiv(
                                lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                rBuffer.baseAddress! as! UnsafePointer<Float>,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData
            } else if T.self == Double.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { rBuffer in
                    rhsData.withUnsafeBufferPointer { lBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vsdivD(
                                lBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                                rBuffer.baseAddress! as! UnsafePointer<Double>,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Double>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData
            } else {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                
                let lDataFloat = lhs.data.map { Float($0) }
                let rDataFloat = rhsData.map { Float($0) }
                
                lDataFloat.withUnsafeBufferPointer { rBuffer in
                    rDataFloat.withUnsafeBufferPointer { lBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vsdiv(
                                lBuffer.baseAddress!, 1,
                                rBuffer.baseAddress!,
                                oBuffer.baseAddress!, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
            }
        
        result.operation = "/"
        result.parents = [
            (lhs, { v in
                let fullSize = [T](repeating: rhs, count: lhs.dataSize)
                return multiply(inverseDivide(fullSize, s: 1), v.gradient!) })
        ]
        return result
    }
}
