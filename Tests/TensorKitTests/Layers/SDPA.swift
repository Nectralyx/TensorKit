//
//  SDPA.swift
//  Nucleus
//
//  Created by Morgan Keay on 2024-10-05.
//

import Foundation
import TensorKit

func SDPA<T: TensorComplex>(_ keys: Tensor<T>, _ queries: Tensor<T>, _ values: Tensor<T>, _ mask: Bool = false, _ ignore_mask: Tensor<T>? = nil) -> Tensor<T> {
    guard keys.shape == queries.shape else {
        fatalError("Keys and Queries must have matching shapes")
    }
    
    let scale = sqrt(1 / sqrt(Tensor(T(queries.shape.last!))))
    var a = (queries * scale) ** (keys.transpose() * scale)

    if mask {
        let mask: Tensor = lowerTriangle(shape: a.shape, upper: T(-Double.infinity), lower: T(0))
        a = a + mask
    }
    
    var attention = Softmax(a, dimension: a.shape.count - 1)
    
    if ignore_mask != nil {
        var mask = ignore_mask!
        if a.shape.count == 3 {
            mask = ignore_mask!.unsqueeze(0).permute([0, 2, 1])
            mask = mask.expand(to: a.shape)
        } else if a.shape.count == 4 {
            mask = mask.unsqueeze(0).unsqueeze(3).permute([1, 0, 2, 3])
            mask = mask.expand(to: a.shape)
        }
        attention = attention * mask
        attention.parents.remove(at: 1)
    }
    
    return attention ** values
}

