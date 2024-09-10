//
//  File.metal
//  
//
//  Created by Morgan Keay on 2024-09-09.
//

#include <metal_stdlib>
using namespace metal;

kernel void vectorConcat(
    device const float* vectorA [[buffer(0)]],  // First input vector
    device const float* vectorB [[buffer(1)]],  // Second input vector
    device float* result [[buffer(2)]],         // Output vector for concatenation
    constant uint& sizeA [[buffer(3)]],         // Size of the first vector
    constant uint& sizeB [[buffer(4)]],         // Size of the second vector
    uint id [[thread_position_in_grid]]         // Thread ID
) {
    // First copy elements from vectorA to result
    if (id < sizeA) {
        result[id] = vectorA[id];
    }
    // Then copy elements from vectorB to result after vectorA
    else if (id < sizeA + sizeB) {
        result[id] = vectorB[id - sizeA];
    }
}
