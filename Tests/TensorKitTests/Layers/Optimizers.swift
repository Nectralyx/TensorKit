//
//  Optimizers.swift
//  Synapse
//
// 
//

import Foundation
import Accelerate
import TensorKit

class SGD<T: TensorComplex>: Codable {
    var learningRate: T
    var parameters: [Parameter<T>]
    
    init(learningRate: T, parameters: [Parameter<T>]) {
        self.learningRate = learningRate
        self.parameters = parameters
    }
    func step() {
        for parameter in parameters {
            if T.self == Float.self {
                parameter.gradient!.withUnsafeBufferPointer{ grad in
                    parameter.data.withUnsafeMutableBufferPointer{ data in
                        let gradPtr = UnsafeRawPointer(grad.baseAddress!).assumingMemoryBound(to: Float.self)
                        let dataPtr = UnsafeMutableRawPointer(data.baseAddress!).assumingMemoryBound(to: Float.self)
                        let negLearningRate = [-learningRate as! Float]
                        vDSP_vsma(gradPtr,
                                  1,
                                  negLearningRate,
                                  dataPtr,
                                  1,
                                  dataPtr,
                                  1,
                                  vDSP_Length(data.count))
                    }
                }
            } else if T.self == Double.self {
                parameter.gradient!.withUnsafeBufferPointer{ grad in
                    parameter.data.withUnsafeMutableBufferPointer{ data in
                        let gradPtr = UnsafeRawPointer(grad.baseAddress!).assumingMemoryBound(to: Double.self)
                        let dataPtr = UnsafeMutableRawPointer(data.baseAddress!).assumingMemoryBound(to: Double.self)
                        let negLearningRate = [-learningRate as! Double]
                        vDSP_vsmaD(gradPtr,
                                  1,
                                  negLearningRate,
                                  dataPtr,
                                  1,
                                  dataPtr,
                                  1,
                                  vDSP_Length(data.count))
                    }
                }
            } else {
                parameter.data = zip(parameter.data, parameter.gradient!).map{ $0.0 - $0.1 * learningRate }
            }
        }
    }
    
    func resetGrad() {
        for parameter in parameters {
            let count = vDSP_Length(parameter.gradient!.count)
            if T.self == Float.self {
                parameter.gradient!.withUnsafeMutableBufferPointer{ grad in
                    vDSP_vclr(grad.baseAddress! as! UnsafeMutablePointer<Float>, 1, count)
                }
            } else if T.self == Double.self {
                parameter.gradient!.withUnsafeMutableBufferPointer{ grad in
                    vDSP_vclrD(grad.baseAddress! as! UnsafeMutablePointer<Double>, 1, count)
                }
            } else {
                parameter.gradient = [T](repeating: 0, count: parameter.data.count)
            }
        }
    }
}

class ADAMOptimizer<T: TensorComplex>: Optimizer {
    let beta1: Tensor<T>
    let beta2: Tensor<T>
    let learningRate: Tensor<T>
    var epsilon: Tensor<T> = Tensor<T>(10e-5)
    var parameters: [Parameter<T>]
    var v: [Tensor<T>] = []
    var m: [Tensor<T>] = []
    var t: Int = 0
    init(parameters: [Parameter<T>], beta1: T = 0.9, beta2: T = 0.999, learningRate: T) {
        self.beta1 = Tensor<T>(beta1)
        self.beta2 = Tensor<T>(beta2)
        self.learningRate = Tensor<T>(learningRate)
        self.parameters = parameters
        for parameter in parameters {
            self.v.append(Tensor<T>(.zeros, shape: parameter.shape))
            self.m.append(Tensor<T>(.zeros, shape: parameter.shape))
        }
    }
    
    func step() {
        t += 1
        let one = Tensor<T>(1.0)
        let beta1_t = one - pow(beta1, T(t))
        let beta2_t = one - pow(beta2, T(t))
        for i in 0..<parameters.count {
            let parameter = Tensor(parameters[i].gradient!, shape: parameters[i].shape)
            m[i] = beta1 * m[i] + (one - beta1) * parameter
            v[i] = beta2 * v[i] + (one - beta2) * (parameter * parameter)
            let mhat = m[i] / beta1_t
            let vhat = v[i] / beta2_t
            
            let cap = learningRate * mhat / (sqrt(vhat) + epsilon)

            //parameters[i].data = (parameters[i] - learningRate * mhat / (sqrt(vhat) + epsilon)).data
            var out = (parameters[i] - cap).data
            let threshold: T = 1000
            let norm = T(sqrtf(out.reduce(0) { $0 + Float($1 * $1) }))
            if norm > threshold {
                let scale = threshold / norm
                for i in 0..<out.count {
                    out[i] *= scale
                }
            }
            //parameters[i].data = (parameters[i] - cap).data
            parameters[i].data = out
        }
    }
    
    func resetGrad() {
        for parameter in parameters {
            let count = vDSP_Length(parameter.gradient!.count)
            if T.self == Float.self {
                parameter.gradient!.withUnsafeMutableBufferPointer{ grad in
                    vDSP_vclr(grad.baseAddress! as! UnsafeMutablePointer<Float>, 1, count)
                }
            } else if T.self == Double.self {
                parameter.gradient!.withUnsafeMutableBufferPointer{ grad in
                    vDSP_vclrD(grad.baseAddress! as! UnsafeMutablePointer<Double>, 1, count)
                }
            } else {
                parameter.gradient = [T](repeating: 0, count: parameter.data.count)
            }
        }
    }
}



