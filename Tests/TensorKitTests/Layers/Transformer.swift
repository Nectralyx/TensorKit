//
//  Transformer.swift
//  Nucleus
//
// 
//

import Foundation
import TensorKit

class TransformerEncoderlayer<T: TensorComplex>: Codable {
    var dropRate: T
    let attention: multiHeadAttention<T>
    let linear1: Linear<T>
    let linear2: Linear<T>
    let norm1: LayerNormalization<T>
    let norm2: LayerNormalization<T>
    let parameters: [Parameter<T>]
    init(model_size: Int, feedForwardSize: Int, num_heads: Int, dropRate: T, initializer: TensorInitialization = .xavier_glorot) {
        self.attention = multiHeadAttention(model_size: model_size, num_heads: num_heads, mask: false, initializer: initializer)
        self.linear1 = Linear(inputSize: model_size, outputSize: feedForwardSize, initializer: initializer)
        self.linear2 = Linear(inputSize: feedForwardSize, outputSize: model_size, initializer: initializer)
        self.norm1 = LayerNormalization([model_size])
        self.norm2 = LayerNormalization([model_size])
        self.parameters = attention.parameters + linear1.parameters + linear2.parameters + norm1.parameters + norm2.parameters
        self.dropRate = dropRate
    }

    func sa_block(_ x: Tensor<T>, ignore_mask: Tensor<T>?) -> Tensor<T> {
        let x = attention.forward(x, x, x, ignore_mask)
        return dropOut(x, prob: dropRate)
    }
    func ff_block(_ x: Tensor<T>) -> Tensor<T> {
        let x = linear2.forward(dropOut(ReLU(linear1.forward(x)), prob: dropRate))
        return dropOut(x, prob: dropRate)
    }
    func forward(_ x: Tensor<T>, _ ignore_mask: Tensor<T>? = nil) -> Tensor<T> {
        var x = norm1.forward(
            x + sa_block(x, ignore_mask: ignore_mask)
        )
        x = norm2.forward(
            x + ff_block(x)
        )
        return x
    }
}

class TransformerDecoderLayer<T: TensorComplex>: Codable {
    var dropRate: T
    let selfAttention: multiHeadAttention<T>
    let encoderDecoderAttention: multiHeadAttention<T>
    let linear1: Linear<T>
    let linear2: Linear<T>
    let norm1: LayerNormalization<T>
    let norm2: LayerNormalization<T>
    let norm3: LayerNormalization<T>
    let parameters: [Parameter<T>]
    init(model_size: Int, feedForwardSize: Int, num_heads: Int, dropRate: T, initializer: TensorInitialization = .xavier_glorot) {
        self.dropRate = dropRate
        self.selfAttention = multiHeadAttention(model_size: model_size, num_heads: num_heads, mask: true, initializer: initializer)
        self.encoderDecoderAttention = multiHeadAttention(model_size: model_size, num_heads: num_heads, mask: false, initializer: initializer)
        self.linear1 = Linear(inputSize: model_size, outputSize: feedForwardSize, initializer: initializer)
        self.linear2 = Linear(inputSize: feedForwardSize, outputSize: model_size, initializer: initializer)
        self.norm1 = LayerNormalization([model_size])
        self.norm2 = LayerNormalization([model_size])
        self.norm3 = LayerNormalization([model_size])
        self.parameters = selfAttention.parameters + encoderDecoderAttention.parameters + linear1.parameters + linear2.parameters + norm1.parameters + norm2.parameters + norm3.parameters
    }
    
    func mha_block(_ x: Tensor<T>, _ mem: Tensor<T>, ignore_mask: Tensor<T>?) -> Tensor<T> {
        let x = encoderDecoderAttention.forward(mem, x, mem, ignore_mask)
        return dropOut(x, prob: dropRate)
    }
    
    func sa_block(_ x: Tensor<T>, ignore_mask: Tensor<T>?) -> Tensor<T> {
        let x = selfAttention.forward(x, x, x, ignore_mask)
        return dropOut(x, prob: dropRate)
    }
    func ff_block(_ x: Tensor<T>) -> Tensor<T> {
        let x = linear2.forward(dropOut(ReLU(linear1.forward(x)), prob: dropRate))
        return dropOut(x, prob: dropRate)
    }
    
    func forward(mem: Tensor<T>, tgt: Tensor<T>, mem_ignore_mask: Tensor<T>? = nil, tgt_ignore_mask: Tensor<T>? = nil) -> Tensor<T> {
        var x = tgt
        x = norm1.forward(
            x + sa_block(x, ignore_mask: tgt_ignore_mask)
        )
        x = norm2.forward(
            x + mha_block(x, mem, ignore_mask: mem_ignore_mask)
        )
        x = norm3.forward(x + ff_block(x))
        return x
    }
}

class TransformerDecoder<T: TensorComplex>: Codable {
    var layers: [TransformerDecoderLayer<T>]
    var parameters: [Parameter<T>]
    var norm: LayerNormalization<T>?
    init(_ layer: TransformerDecoderLayer<T>, num_layers: Int, norm: LayerNormalization<T>? = nil) {
        self.layers = [TransformerDecoderLayer<T>](repeating: layer, count: num_layers)
        self.norm = norm
        self.parameters = []
        for i in layers {
            self.parameters.append(contentsOf: i.parameters)
        }
        if norm != nil {
            self.parameters.append(contentsOf: norm!.parameters)
        }
    }
    
    func forward(tgt: Tensor<T>, mem: Tensor<T>, mem_ignore_mask: Tensor<T>?, tgt_ignore_mask: Tensor<T>?) -> Tensor<T> {
        var output = tgt
        for i in layers {
            output = i.forward(mem: mem, tgt: output, mem_ignore_mask: mem_ignore_mask, tgt_ignore_mask: tgt_ignore_mask)
        }
        if norm != nil {
            output = norm!.forward(output)
        }
        return output
    }
}

class TransformerEncoder<T: TensorComplex>: Codable {
    var layers: [TransformerEncoderlayer<T>]
    var parameters: [Parameter<T>]
    var norm: LayerNormalization<T>?
    init(_ layer: TransformerEncoderlayer<T>, num_layers: Int, norm: LayerNormalization<T>? = nil) {
        self.layers = [TransformerEncoderlayer<T>](repeating: layer, count: num_layers)
        self.parameters = layers.flatMap(\.parameters)
        self.norm = norm
        if norm != nil {
            self.parameters.append(contentsOf: norm!.parameters)
        }
    }
    
    func forward(_ src: Tensor<T>, src_ignore_mask: Tensor<T>?) -> Tensor<T> {
        var output = src
        for i in layers {
            output = i.forward(output, src_ignore_mask)
        }
        if norm != nil {
            output = norm!.forward(output)
        }
        return output
    }
}

class Transformer<T: TensorComplex>: Codable {
    let model_size: Int
    let num_heads: Int
    let dropRate: T
    let encoder: TransformerEncoder<T>
    let decoder: TransformerDecoder<T>
    let parameters: [Parameter<T>]
    
    init(model_size: Int, num_heads: Int, num_encoder_layers: Int, num_decoder_layers: Int, feedForwardSize: Int, dropRate: T, initializer: TensorInitialization = .xavier_glorot) {
        let encoderLayer = TransformerEncoderlayer(
            model_size: model_size,
            feedForwardSize: feedForwardSize,
            num_heads: num_heads,
            dropRate: dropRate,
            initializer: initializer
        )
        let encoderNorm = LayerNormalization<T>([model_size])
        self.encoder = TransformerEncoder(
            encoderLayer,
            num_layers: num_encoder_layers,
            norm: encoderNorm
        )
        
        let decoderLayer = TransformerDecoderLayer(
            model_size: model_size,
            feedForwardSize: feedForwardSize,
            num_heads: num_heads,
            dropRate: dropRate,
            initializer: initializer
        )
        let decoderNorm = LayerNormalization<T>([model_size])
        self.decoder = TransformerDecoder(
            decoderLayer,
            num_layers: num_decoder_layers,
            norm: decoderNorm
        )
        
        self.parameters = encoder.parameters + decoder.parameters
        self.model_size = model_size
        self.num_heads = num_heads
        self.dropRate = dropRate
    }
    
    func forward(src: Tensor<T>, tgt: Tensor<T>, src_mask: Tensor<T>?, tgt_mask: Tensor<T>?) -> Tensor<T> {
        let memory = encoder.forward(
            src,
            src_ignore_mask: src_mask
        )
        let output = decoder.forward(
            tgt: tgt,
            mem: memory,
            mem_ignore_mask: src_mask,
            tgt_ignore_mask: tgt_mask
        )
        return output
    }
}

class StandardTransformer<T: TensorComplex>: Codable {
    var dropRate: T
    let embeddings: Embedding<T>
    let linear: Linear<T>
    var model: Transformer<T>
    var parameters: [Parameter<T>]
    var training: Bool = true
    let sos: Int
    let eos: Int
    let pad: Int
    let embeddingScale: Tensor<T>
    let seq_length: Int
    let num_embeddings: Int
    init(model_size: Int, num_Embeddings: Int, seq_length: Int, num_heads: Int, num_layers: Int, dropRate: T, eos: Int, pad: Int, sos: Int, initializer: TensorInitialization = .xavier_glorot) {
        self.seq_length = seq_length
        self.eos = eos
        self.sos = sos
        self.pad = pad
        self.dropRate = dropRate
        self.linear = Linear(inputSize: model_size, outputSize: num_Embeddings, initializer: initializer)
        self.embeddings = Embedding(embeddings: num_Embeddings, dimensions: model_size, initializer: .random_small)
        self.embeddingScale = sqrt(Tensor<T>(T(model_size)))
        self.model = Transformer(
            model_size: model_size,
            num_heads: num_heads,
            num_encoder_layers: num_layers,
            num_decoder_layers: num_layers,
            feedForwardSize: model_size * 4,
            dropRate: dropRate,
            initializer: initializer
        )
        self.num_embeddings = num_Embeddings
        self.parameters = linear.parameters + embeddings.parameters + model.parameters
    }
    
    func forward(_ src: Tensor<T>, _ tgt: Tensor<T>) -> Tensor<T> {
        var intgt = Tensor(tgt)
        var src = src
        
        let tgt_mask = Tensor<T>(intgt.data.map{ $0 == T(pad) ? T(0) : T(1) }, shape: intgt.shape)
        let src_mask = Tensor<T>(src.data.map{ $0 == T(pad) ? T(0) : T(1) }, shape: src.shape)
        
        src = embeddings.forward(src) * embeddingScale
        positionalEncoding(&src)
        intgt = embeddings.forward(intgt) * embeddingScale
        positionalEncoding(&intgt)

        var output = model.forward(src: src, tgt: intgt, src_mask: src_mask, tgt_mask: tgt_mask)
        output = linear.forward(output)

        return output
    }
    
    func forward(_ src: Tensor<T>) -> Tensor<T> {
        var ointgt = Tensor<T>([T](repeating: T(pad), count: seq_length), shape: [1, seq_length])
        var src = src
        let strides = generateStrides(ointgt.shape)
        /*for i in 0..<ointgt.shape.first! {
            let idx = strides[0] * i
            ointgt.data.insert(T(sos), at: idx)
            _ = ointgt.data.remove(at: strides[1] * (i + 1))
        }*/
        ointgt.data.insert(T(sos), at: 0)
        _ = ointgt.data.dropLast()
        ointgt = ointgt.expand(to: src.shape.dropLast() + [seq_length])
        //print("intgt: \(ointgt)")
        
        let src_mask = Tensor<T>(src.data.map{ $0 == T(pad) ? T(0) : T(1) }, shape: src.shape)
        
        src = embeddings.forward(src) * embeddingScale
        positionalEncoding(&src)
        
        for i in 1..<seq_length {
            
            let tgt_mask = Tensor<T>(ointgt.data.map{ $0 == T(pad) ? T(0) : T(1) }, shape: ointgt.shape)
            
            var intgt = embeddings.forward(ointgt) * embeddingScale
            positionalEncoding(&intgt)
            
            //print("Masks: ")
            //print(src_mask)
            //print(tgt_mask)
            var output = model.forward(src: src, tgt: intgt, src_mask: src_mask, tgt_mask: tgt_mask)
            output = linear.forward(output)
            let final = Softmax(output, dimension: output.shape.count - 1)
            //print("Final: ")
            //print(final)
            let stride = generateStrides(final.shape)
            let sample = final.data[((i - 1) * stride[stride.count - 2])..<(i) * stride[stride.count - 2]]
            //let generated = greedyEncoding(from: Array(sample))
            let generated = final.argmax(final.shape.count - 1)
            let isolated = generated.isolate(generated.shape.count - 2, i - 1)
            //print("Isolated: \(isolated)")
            for j in 0..<ointgt.dataSize / seq_length {
                let idx = strides[0] * (j) + i
                ointgt.data[idx] = isolated.data[j]
                
            }
            
            if ointgt.shape.count == 2 && ointgt.shape[0] == 1 {
                if generated.data[0] == T(eos) {
                    break
                }
            }
            
            //print("Generated: \(generated)")
            //print(ointgt)
        }
        //print(ointgt)
        return ointgt
    }
}

func weightedRandomSelection<T: TensorComplex>(from values: [T]) -> T? {
    guard !values.isEmpty else {
        return nil // Return nil if the array is empty
    }
    
    // Calculate the total weight by summing the values
    let totalWeight = values.reduce(0, { $0 + Double(truncating: $1 as! NSNumber) })
    
    // Generate a random number between 0 and totalWeight
    let randomValue = Double.random(in: 0..<totalWeight)
    
    // Find the selected value based on the random number
    var cumulativeWeight: Double = 0.0
    for (idx, value) in values.enumerated() {
        cumulativeWeight += Double(truncating: value as! NSNumber)
        if randomValue < cumulativeWeight {
            return T(idx) // Return the corresponding value
        }
    }
    
    return nil // This should not happen if values are non-empty
}

func greedyEncoding<T: TensorComplex>(from values: [T]) -> T? {
    guard !values.isEmpty else {
        return nil
    }
    
    var maxIndex = 0
    var maxValue = T(-Float.infinity)
    for (index, i) in values.enumerated() {
        if i > maxValue {
            maxIndex = index
            maxValue = i
        }
    }
    return T(maxIndex)
}
