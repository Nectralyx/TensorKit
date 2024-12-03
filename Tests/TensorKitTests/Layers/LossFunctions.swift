//
//  LossFunctions.swift
//  Synapse
//
//  
//

import Foundation
import TensorKit
import TKCore

func meanSquaredError<T: TensorComplex>(predictions: Tensor<T>, targets: Tensor<T>) -> Tensor<T> {
    var output = Tensor<T>(.empty, shape: predictions.shape)
    let n = Tensor<T>([1.0 / T(predictions.dataSize)], shape: [1], calculate_grad: false)
    
    output = targets - predictions
    output = pow(output, 2)
    output = output * n
    return output.sum()
}

enum Reduction {
    case mean
    case sum
    case none
}

func crossEntropy<T: TensorComplex>(predictions: Tensor<T>, targets: Tensor<T>, reduction: Reduction = .mean, ignore_index: [T] = [], print_loss: Bool = true) -> Tensor<T> {

    if predictions.shape.count == 1 {
        predictions.shape.insert(1, at: 0)
        targets.shape.insert(1, at: 0)
    }
    
    guard predictions.shape.dropLast().last! == targets.shape.last! else {
        fatalError("Shapes of predictions and targets must match, and must be less than 3. Prediction shape: \(predictions.shape), Target shape: \(targets.shape).")
    }
    if print_loss {
        let exexpon = exp(predictions)
        let denominator = exexpon.sum(along: exexpon.shape.count - 1, keepDims: false)
        let indicesForCorrect: [[Int]] = targets.data.enumerated().map{ [$0.offset, Int($0.element)] }
        let correctValues = predictions.select(indicesForCorrect)
        let indexMap = Tensor<T>(targets.data.map{ ignore_index.contains($0) ? 0 : -1 }, shape: targets.shape)
        let gradientIndexMap = predictions.shape.count < 3 ? Tensor<T>(targets.data.map{ ignore_index.contains($0) ? 0 : 1 }, shape: targets.shape).transpose().expand(to: predictions.shape) : Tensor<T>(targets.data.map{ ignore_index.contains($0) ? 0 : 1}, shape: targets.shape).unsqueeze(1).transpose().expand(to: predictions.shape)
        let indicatorMap = Tensor<T>(.zeros, shape: predictions.shape)
        var ignored = indexMap.data.filter{ $0 == 0 }.count
        ignored = targets.dataSize - Int(ignored)
        let strides = generateStrides(indicatorMap.shape)
        for i in 0..<indicatorMap.dataSize / indicatorMap.shape[indicatorMap.shape.count - 1] {
            indicatorMap.data[strides[indicatorMap.shape.count - 2] * i + Int(targets.data[i])] = -1
        }
        var logs = (correctValues - log(denominator))
        logs = logs * indexMap
        logs.parents = [
            (predictions, { v in
                return (((Softmax(predictions, dimension: predictions.shape.count - 1) + indicatorMap) * gradientIndexMap) / Tensor(T(ignored))).data
            })
        ]
        logs.operation = "Cross Entropy Loss"
        if reduction == .none {
            return logs
        } else if reduction == .sum {
            return logs.sum(along: 0).sum()
        } else {
            return logs.sum() / Tensor<T>(T(ignored))
        }
    } else {
        let logs = Tensor<T>(0)
        
        let indexMap = Tensor<T>(targets.data.map{ ignore_index.contains($0) ? 0 : -1 }, shape: targets.shape)
        let gradientIndexMap = Tensor<T>(targets.data.map{ ignore_index.contains($0) ? 0 : 1 }, shape: targets.shape).transpose().expand(to: predictions.shape)
        let indicatorMap = Tensor<T>(.zeros, shape: predictions.shape)
        var ignored = indexMap.data.filter{ $0 == 0 }.count
        ignored = targets.dataSize - Int(ignored)
        let strides = generateStrides(indicatorMap.shape)
        for i in 0..<indicatorMap.shape[0] {
            indicatorMap.data[strides[0] * i + Int(targets.data[i])] = -1
        }
        
        logs.parents = [
            (predictions, { v in
                return (((Softmax(predictions, dimension: 1) + indicatorMap) * gradientIndexMap) / Tensor(T(ignored))).data
            })
        ]
        logs.operation = "Cross Entropy Optimized Loss"
        return logs
    }
}

