//
//  Use this file to import your target's public headers that you would like to expose to Swift.
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

#pragma once

//#include <classImpl.h>

void testing(int a, int result) {
    result = a;
}

#include <stdio.h>

template <typename T>
void vvadd(const T* a, const T* b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

template <typename T>
void vvsubtract(const T* a, const T* b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[i];
    }
}

template <typename T>
void vvmultiply(const T* a, const T* b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

template <typename T>
void vvdivide(const T* a, const T* b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] / b[i];
    }
}

// Explicit instantiation declarations
template void vvadd<float>(const float* a, const float* b, float* result, int size);
template void vvsubtract<float>(const float* a, const float* b, float* result, int size);
template void vvmultiply<float>(const float* a, const float* b, float* result, int size);
template void vvdivide<float>(const float* a, const float* b, float* result, int size);

template void vvadd<double>(const double* a, const double* b, double* result, int size);
template void vvsubtract<double>(const double* a, const double* b, double* result, int size);
template void vvmultiply<double>(const double* a, const double* b, double* result, int size);
template void vvdivide<double>(const double* a, const double* b, double* result, int size);

template <typename T>
void vsadd(const T* a, const T b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b;
    }
}

template <typename T>
void vssubtract(const T* a, const T b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b;
    }
}

template <typename T>
void svsubtract(const T a, const T* b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a - b[i];
    }
}


template <typename T>
void vsmultiply(const T* a, const T b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b;
    }
}

template <typename T>
void vsdivide(const T* a, const T b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] / b;
    }
}

template <typename T>
void svdivide(const T a, const T* b, T* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a / b[i];
    }
}

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

template <typename T>
void sum(const T* a, T result, int size) {
    for (int i = 0; i < size; i++) {
        result += a[i];
    }
}
template <typename T>
void sum(const T* a, int along, T* result, const int* shape, int size, int numBlocks, int blockStride) {
    int dimsize = shape[along];
    
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * dimsize) + (blockIndex % blockStride);
        
        for (int j = 0; j <dimsize; j++) {
            int index = blockStartIndex + j * blockStride;
            result[blockIndex] += a[index];
        }
    }
    
}




// Explicit instantiation declarations
template void vsadd<float>(const float* a, const float b, float* result, int size);
template void vssubtract<float>(const float* a, const float b, float* result, int size);
template void svsubtract<float>(const float a, const float* b, float* result, int size);
template void vsmultiply<float>(const float* a, const float b, float* result, int size);
template void vsdivide<float>(const float* a, const float b, float* result, int size);
template void svdivide<float>(const float a, const float* b, float* result, int size);

template void vsadd<double>(const double* a, const double b, double* result, int size);
template void vssubtract<double>(const double* a, const double b, double* result, int size);
template void svsubtract<double>(const double a, const double* b, double* result, int size);
template void vsmultiply<double>(const double* a, const double b, double* result, int size);
template void vsdivide<double>(const double* a, const double b, double* result, int size);
template void svdivide<double>(const double a, const double* b, double* result, int size);

template void upperTriangle(int rows, int cols, float upper, float lower, float* result);
template void lowerTriangle(int rows, int cols, float upper, float lower, float* result);

template void upperTriangle(int rows, int cols, double upper, double lower, double* result);
template void lowerTriangle(int rows, int cols, double upper, double lower, double* result);

template void sum(const float* a, float result, int size);
template void sum(const float* a, int along, float* result, const int* shape, int size, int numBlocks, int blockStride);

template void sum(const double* a, double result, int size);
template void sum(const double* a, int along, double* result, const int* shape, int size, int numBlocks, int blockStride);

