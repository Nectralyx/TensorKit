//
//  PositionalEncoder.swift
//  Synapse
//
//  
//

import Foundation
import TensorKit
import TKCore

func positionalEncoding<T: TensorComplex>(_ x: inout Tensor<T>) {
    let d_model = x.shape.last!
    let positions = x.shape.dropLast().reduce(1, *)
    x.data.withUnsafeMutableBufferPointer { ptr in
        posEnc(
            ptr.baseAddress! as? UnsafeMutablePointer<Float>,
            Int32(d_model),
            Int32(positions)
        )
    }
}

