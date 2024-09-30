//
//  Generation Functions.h
//  Synapse
//
//  Created by Morgan Keay on 2024-09-15.
//

#ifndef Generation_Functions_h
#define Generation_Functions_h

#include <stdio.h>

template <typename T>
void upperTriangle(int rows, int cols, T upper, T lower, T* result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T fill = 0;
            if (j >= i) {
                fill = upper;
            } else {
                fill = lower;
            }
            result[i * cols + j] = fill;
        }
    }
}

template <typename T>
void lowerTriangle(int rows, int cols, T upper, T lower, T* result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T fill = 0;
            if (i >= j) {
                fill = lower;
            } else {
                fill = upper;
            }
            result[i * cols + j] = fill;
        }
    }
}

#include <cstring>  // For std::memcpy
#include <algorithm> //For std::min

void repeatArray(const float* input, float* output, size_t inputSize, size_t repeatCount) {
    size_t totalSize = inputSize * repeatCount;

    // Copy the input array to the output array once
    std::memcpy(output, input, inputSize * sizeof(float));

    // Use loop unrolling to copy blocks of data efficiently
    size_t currentSize = inputSize;
    while (currentSize < totalSize) {
        size_t copySize = std::min(currentSize, totalSize - currentSize);
        std::memcpy(output + currentSize, output, copySize * sizeof(float));
        currentSize += copySize;
    }
}


template void upperTriangle(int rows, int cols, float upper, float lower, float* result);
template void lowerTriangle(int rows, int cols, float upper, float lower, float* result);

template void upperTriangle(int rows, int cols, double upper, double lower, double* result);
template void lowerTriangle(int rows, int cols, double upper, double lower, double* result);

#endif /* Generation_Functions_h */
