//
//  Use this file to import your target's public headers that you would like to expose to Swift.
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

#pragma once

//#include <classImpl.h>
#include <stdio.h>
#include <cstring>  // For std::memcpy
#include <iostream>
#include <algorithm> //For std::min
/*
static void vvadd(const float* a, const float* b, float* result, int size) {
    for (int i = 0; i < size; i += 2) {
        result[i] = a[i] + b[i];
        result[i + 1] = a[i + 1] + b[i + 1];
    }
}*/
/*
static void vvadd(const float* a, const float* b, float* result, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}*/
// Example using GCC intrinsics for non-temporal stores (for unaligned data)
#include <arm_neon.h>

/*
static void vvadd(const float* a, const float* b, float* result, int size) {
    int i = 0;
    for (; i < size - 4; i += 4) {
        result[i] = a[i] + b[i];
        result[i + 1] = a[i + 1] + b[i + 1];
        result[i + 2] = a[i + 2] + b[i + 2];
        result[i + 3] = a[i + 3] + b[i + 3];
    }
    for (; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}
*/
#include <arm_neon.h>

static void vvadd(const float* a, const float* b, float* result, int size) {
    int simdSize = size / 4 * 4;  // Process in chunks of 4 floats
#pragma omp parallel for
    for (int i = 0; i < simdSize; i += 4) {
        float32x4_t av = vld1q_f32(&a[i]);  // Load 4 floats from a
        float32x4_t bv = vld1q_f32(&b[i]);  // Load 4 floats from b
        float32x4_t rv = vaddq_f32(av, bv); // Add the 4 floats
        vst1q_f32(&result[i], rv);          // Store the result
    }
    // Handle the remaining elements
    for (int i = simdSize; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}
 

static void vvsubtract(const float* a, const float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[i];
    }
}

static void vvmultiply(const float* a, const float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

static void matrixmultiply(const float* a, const float* b, const int m, const int n, const int p, float* result) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                //result[i * m + j] += a[i * m + k] * b[k * n + j];
                result[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
}

static void vvdivide(const float* a, const float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] / b[i];
    }
}

static void vvaddD(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

static void vvsubtractD(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[i];
    }
}

static void vvmultiplyD(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

static void matrixmultiplyD(const double* a, const double* b, const int m, const int n, const int p, double* result) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                //result[i * m + j] += a[i * m + k] * b[k * n + j];
                result[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
}

static void vvdivideD(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] / b[i];
    }
}


static void vsadd(const float* a, const float b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b;
    }
}


static void vssubtract(const float* a, const float b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b;
    }
}


static void svsubtract(const float a, const float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a - b[i];
    }
}



static void vsmultiply(const float* a, const float b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b;
    }
}


static void vsdivide(const float* a, const float b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] / b;
    }
}


static void svdivide(const float a, const float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a / b[i];
    }
}

static void vsaddD(const double* a, const double b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b;
    }
}


static void vssubtractD(const double* a, const double b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b;
    }
}


static void svsubtractD(const double a, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a - b[i];
    }
}



static void vsmultiplyD(const double* a, const double b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b;
    }
}


static void vsdivideD(const double* a, const double b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] / b;
    }
}


static void svdivideD(const double a, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a / b[i];
    }
}

static void upperTriangle(int rows, int cols, float upper, float lower, float* result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float fill = 0;
            if (j >= i) {
                fill = upper;
            } else {
                fill = lower;
            }
            result[i * cols + j] = fill;
        }
    }
}

static void lowerTriangle(int rows, int cols, float upper, float lower, float* result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float fill = 0;
            if (i >= j) {
                fill = lower;
            } else {
                fill = upper;
            }
            result[i * cols + j] = fill;
        }
    }
}

static void upperTriangleD(int rows, int cols, double upper, double lower, double* result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double fill = 0;
            if (j >= i) {
                fill = upper;
            } else {
                fill = lower;
            }
            result[i * cols + j] = fill;
        }
    }
}

static void lowerTriangleD(int rows, int cols, double upper, double lower, double* result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double fill = 0;
            if (i >= j) {
                fill = lower;
            } else {
                fill = upper;
            }
            result[i * cols + j] = fill;
        }
    }
}

static void repeatArray(const float* input, float* output, size_t inputSize, size_t repeatCount) {
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

static void repeatArrayD(const double* input, double* output, size_t inputSize, size_t repeatCount) {
    size_t totalSize = inputSize * repeatCount;

    // Copy the input array to the output array once
    std::memcpy(output, input, inputSize * sizeof(double));

    // Use loop unrolling to copy blocks of data efficiently
    size_t currentSize = inputSize;
    while (currentSize < totalSize) {
        size_t copySize = std::min(currentSize, totalSize - currentSize);
        std::memcpy(output + currentSize, output, copySize * sizeof(double));
        currentSize += copySize;
    }
}


static void sum(const float* a, float result, int size) {
    for (int i = 0; i < size; i++) {
        result += a[i];
    }
}

static void sum(const float* a, int along, float* result, const int* shape, int size, int numBlocks, int blockStride) {
    int dimsize = shape[along];
    
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * dimsize) + (blockIndex % blockStride);
        
        for (int j = 0; j <dimsize; j++) {
            int index = blockStartIndex + j * blockStride;
            result[blockIndex] += a[index];
        }
    }
}

static void sumD(const double* a, double result, int size) {
    for (int i = 0; i < size; i++) {
        result += a[i];
    }
}

static void sumD(const double* a, int along, double* result, const int* shape, int size, int numBlocks, int blockStride) {
    int dimsize = shape[along];
    
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * dimsize) + (blockIndex % blockStride);
        
        for (int j = 0; j <dimsize; j++) {
            int index = blockStartIndex + j * blockStride;
            result[blockIndex] += a[index];
        }
    }
}
