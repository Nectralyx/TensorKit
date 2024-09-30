//
//  -.swift
//  Synapse
//
//
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import Foundation
import TKCore

public extension Tensor {
    @inlinable
    static func -(lhs: Tensor, rhs: Tensor) -> Tensor {
        let finalShape = mergeShapes(lhs.shape, rhs.shape)
        let lhs = lhs.expand(to: finalShape)
        let rhs = rhs.expand(to: finalShape)
        let result = Tensor(.empty, shape: finalShape, calculate_grad: (lhs.gradient != nil || rhs.gradient != nil) ? true : false)
        
            if T.self == Float.self {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { lBuffer in
                    rhs.data.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vvsubtract(
                                rBuffer.baseAddress! as? UnsafePointer<Float>,
                                lBuffer.baseAddress! as? UnsafePointer<Float>,
                                oBuffer.baseAddress!,
                                Int32(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData as! [T]
            } else if T.self == Double.self {
                var outputData = [Double](repeating: 0, count: result.dataSize)
                
                lhs.data.withUnsafeBufferPointer { lBuffer in
                    rhs.data.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vvsubtractD(
                                rBuffer.baseAddress! as? UnsafePointer<Double>,
                                lBuffer.baseAddress! as? UnsafePointer<Double>,
                                oBuffer.baseAddress!,
                                Int32(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData as! [T]
            } else /*if T.self == Float16.self*/ {
                var outputData = [Float](repeating: 0, count: result.dataSize)
                
                let lDataFloat = lhs.data.compactMap { Float($0) }
                let rDataFloat = rhs.data.compactMap { Float($0) }
                
                lDataFloat.withUnsafeBufferPointer { lBuffer in
                    rDataFloat.withUnsafeBufferPointer { rBuffer in
                        outputData.withUnsafeMutableBufferPointer { oBuffer in
                            vvsubtract(
                                rBuffer.baseAddress!,
                                lBuffer.baseAddress!,
                                oBuffer.baseAddress!,
                                Int32(result.dataSize)
                            )
                        }
                    }
                }
                
                result.data = outputData.compactMap { T($0) }
            }
        
        result.operation = "-"
        result.parents = [
            (lhs, { v in v.gradient! }),
            (rhs, { v in multiply(v.gradient!, s: -1.0) })
        ]
        
        return result
    }

}

