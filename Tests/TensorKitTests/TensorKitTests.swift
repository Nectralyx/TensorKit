/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

import XCTest
import TensorKit

final class TensorKitTests: XCTestCase {
    
    func testTensorPerformance() throws {
        //LEGEND
        // 3: <sos>
        // 4: <eos>
        // 5: <pad>
        
        let input = Tensor<Float>([
            [0, 1, 2, 4, 5]
        ], calculate_grad: false)
        let target = Tensor<Float>([
            [3, 2, 1, 0, 4]
        ], calculate_grad: false)
        let lossTarget = Tensor<Float>([
            [2, 1, 0, 4, 5]
        ])
        let learningRate = 1e-3
        let transformer = StandardTransformer<Float>(
            model_size: 512,
            num_Embeddings: 6,
            seq_length: 5,
            num_heads: 8,
            num_layers: 6,
            dropRate: 0.0,
            eos: 4,
            pad: 5,
            sos: 3,
            initializer: .xavier_glorot
        )
        
        let optim = ADAMOptimizer(parameters: transformer.parameters, learningRate: Float(learningRate))
        
        transformer.training = true
        
        for i in 0..<500 {
            let output = transformer.forward(input, target).squeeze()
            let loss = crossEntropy(predictions: output, targets: lossTarget, ignore_index: [5])
            loss.backward(printSteps: false)
            print("Epoch: \(i) | Loss: \(loss)")
            optim.step()
            optim.resetGrad()
        }
        print(transformer.forward(input))
    }
    // If the operations were optimized properly and leaks didn't occur, this function *should* run in < 1 minute, and require small amounts of memory.
}
