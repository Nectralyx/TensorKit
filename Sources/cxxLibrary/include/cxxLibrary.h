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
#include <cstring>  // For std::memcpy
#include <algorithm> //For std::min

static void vvadd(const float* a, const float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
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
