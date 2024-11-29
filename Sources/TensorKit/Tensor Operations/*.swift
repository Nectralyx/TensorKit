//
//  *.swift
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
    static func *(lhs: Tensor, rhs: Tensor) -> Tensor {
        let finalShape = mergeShapes(lhs.shape, rhs.shape)
        let lhs = lhs.expand(to: finalShape)
        let rhs = rhs.expand(to: finalShape)
        let result = Tensor(.empty, shape: finalShape, calculate_grad: (lhs.gradient != nil || rhs.gradient != nil) ? true : false)
        
            if T.self == Float.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { lBuffer in
                    rhs.data.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vmul(
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
                
                lhs.data.withUnsafeBufferPointer { lBuffer in
                    rhs.data.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vmulD(
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
                
                let lDataFloat = lhs.data.compactMap { Float($0) }
                let rDataFloat = rhs.data.compactMap { Float($0) }
                
                lDataFloat.withUnsafeBufferPointer { lBuffer in
                    rDataFloat.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vmul(
                                lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
            }
        
        result.operation = "*"
        result.parents = [
            (lhs, { v in multiply(rhs.data, v.gradient!) }),
            (rhs, { v in multiply(lhs.data, v.gradient!) })
        ]
        return result
    }
    
    @inlinable
    static func *(lhs: T, rhs: Tensor) -> Tensor {
        let finalShape = rhs.shape
        let lhsData = [lhs]
        let result = Tensor(.empty, shape: finalShape, calculate_grad: rhs.gradient != nil)
        
            if T.self == Float.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                
                lhsData.withUnsafeBufferPointer { lBuffer in
                    rhs.data.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vsmul(
                                rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                lBuffer.baseAddress! as! UnsafePointer<Float>,
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
                            vDSP_vsmulD(
                                rBuffer.baseAddress! as! UnsafePointer<Double>, 1,
                                lBuffer.baseAddress! as! UnsafePointer<Double>,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Double>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData
            } else {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                
                let lDataFloat = lhsData.compactMap { Float($0) }
                let rDataFloat = rhs.data.compactMap { Float($0) }
                
                lDataFloat.withUnsafeBufferPointer { lBuffer in
                    rDataFloat.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vsmul(
                                rBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                lBuffer.baseAddress! as! UnsafePointer<Float>,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
            }
        
        result.operation = "*"
        result.parents = [
            (rhs, { v in
                let fullSize = [T](repeating: lhs, count: rhs.dataSize)
                return multiply(fullSize, v.gradient!)
            })
        ]
        return result
    }
    
    @inlinable
    static func *(lhs: Tensor, rhs: T) -> Tensor {
        let finalShape = lhs.shape
        let rhsData = [rhs]
        let result = Tensor(.empty, shape: finalShape, calculate_grad: lhs.gradient != nil)
        
            if T.self == Float.self {
                var outputData = [T](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { lBuffer in
                    rhsData.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vsmul(
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
                
                lhs.data.withUnsafeBufferPointer { lBuffer in
                    rhsData.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vsmulD(
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
                
                let lDataFloat = lhs.data.compactMap { Float($0) }
                let rDataFloat = rhsData.compactMap { Float($0) }
                
                lDataFloat.withUnsafeBufferPointer { lBuffer in
                    rDataFloat.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vDSP_vsmul(
                                lBuffer.baseAddress! as! UnsafePointer<Float>, 1,
                                rBuffer.baseAddress! as! UnsafePointer<Float>,
                                oBuffer.baseAddress! as! UnsafeMutablePointer<Float>, 1,
                                vDSP_Length(result.dataSize)
                            )
                        }
                    }
                }
            }
        
        result.operation = "*"
        result.parents = [
            (lhs, { v in
                let fullSize = [T](repeating: rhs, count: lhs.dataSize)
                return multiply(fullSize, v.gradient!)
            })
        ]
        return result
    }
}
