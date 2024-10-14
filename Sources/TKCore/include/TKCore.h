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
#include <cmath>
#include <numeric>
#include <vector>
#include <limits>
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

float* matrixMultiply(const float* a, const float* b, const int* aShape, const int* bShape) {
    // Matrix dimensions
    int aRows = aShape[0];
    int aCols = aShape[1];
    int bRows = bShape[0];
    int bCols = bShape[1];

    // Ensure that the inner dimensions match (aCols == bRows)
    if (aCols != bRows) {
        std::cerr << "Matrix dimensions do not match for multiplication!" << std::endl;
        return nullptr;  // Return null pointer on error
    }

    // Allocate memory for the result matrix (aRows x bCols)
    float* result = new float[aRows * bCols];

    // Initialize result matrix to zero
    for (int i = 0; i < aRows * bCols; i++) {
        result[i] = 0.0f;
    }

    // Perform matrix multiplication
    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < bCols; ++j) {
            for (int k = 0; k < aCols; ++k) {
                result[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
            }
        }
    }

    return result;
}

double* matrixMultiplyD(const double* a, const double* b, const int* aShape, const int* bShape) {
    // Matrix dimensions
    int aRows = aShape[0];
    int aCols = aShape[1];
    int bRows = bShape[0];
    int bCols = bShape[1];

    // Ensure that the inner dimensions match (aCols == bRows)
    if (aCols != bRows) {
        std::cerr << "Matrix dimensions do not match for multiplication!" << std::endl;
        return nullptr;  // Return null pointer on error
    }

    // Allocate memory for the result matrix (aRows x bCols)
    double* result = new double[aRows * bCols];

    // Initialize result matrix to zero
    for (int i = 0; i < aRows * bCols; i++) {
        result[i] = 0.0f;
    }

    // Perform matrix multiplication
    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < bCols; ++j) {
            for (int k = 0; k < aCols; ++k) {
                result[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
            }
        }
    }

    return result;
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

static void posEnc(float* x, const int d_model, const int positions) {
    for (int pos = 0; pos < positions; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = pos / powf(10000.0f, (2.0f * (float)(i / 2)) / (float)d_model);
            // Apply sinf to even indices and cosf to odd indices
            if (i % 2 == 0) {
                x[pos * d_model + i] = sinf(angle); // Apply sine to even indices
            } else {
                x[pos * d_model + i] = cosf(angle); // Apply cosine to odd indices
            }
        }
    }
}

static void softmax(const float* x, float* y, const int* shape, const int dim, const int dimSize, const int numBlocks, const int blockStride) {
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * dimSize) + (blockIndex % blockStride);
        
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < dimSize; j++) {
            int index = blockStartIndex + j * blockStride;
            max_val = std::max(max_val, x[index]);
        }
        
        float expSum = 0.0f;
        for (int j = 0; j < dimSize; j++) {
            int index = blockStartIndex + j * blockStride;
            float shiftedValue = x[index] - max_val;
            y[index] = std::exp(x[index] - max_val);
            expSum += y[index];
        }
        
        for (int j = 0; j < dimSize; j++) {
            int index = blockStartIndex + j * blockStride;
            y[index] /= expSum;
        }
    }
}

static void softmaxD(const double* x, double* y, const int* shape, const int dim, const int dimSize, const int numBlocks, const int blockStride) {
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * dimSize) + (blockIndex % blockStride);
        
        double max_val = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < dimSize; j++) {
            int index = blockStartIndex + j * blockStride;
            max_val = std::max(max_val, x[index]);
        }
        
        double expSum = 0.0;
        for (int j = 0; j < dimSize; j++) {
            int index = blockStartIndex + j * blockStride;
            double shiftedValue = x[index] - max_val;
            y[index] = std::exp(x[index] - max_val);
            expSum += y[index];
        }
        
        for (int j = 0; j < dimSize; j++) {
            int index = blockStartIndex + j * blockStride;
            y[index] /= expSum;
        }
    }
}

static void softmaxJacobian(const float* x, float* y, const float* outputGrad, const int dimension, const int* shape, const int jSize, const int dataSize, const int blockStride) {
    int numBlocks = dataSize / jSize;
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride);
        
        float* slice = new float[jSize];
        float* outputSlice = new float[jSize];
        
        for (int i = 0; i < jSize; i++) {
            int index = blockStartIndex + i * blockStride;
            slice[i] = x[index];
            outputSlice[i] = outputGrad[index];
        }
        
        float* jacobian = new float[jSize * jSize];
        
        for (int i = 0; i < jSize; i++) {
            for (int j = 0; j < jSize; j++) {
                jacobian[i * jSize + j] = (i == j) ? slice[i] * (1 - slice[i]) : -(slice[i] * slice[j]);
            }
        }
        int aShape[] = {jSize, jSize};
        int bShape[] = {jSize, 1};
        float* output = matrixMultiply(jacobian, outputSlice, aShape, bShape);
        for (int i = 0; i < jSize; i++) {
            int index = blockStartIndex + i * blockStride;
            y[index] = output[i];
        }
    }
}

static void softmaxJacobianD(const double* x, double* y, const double* outputGrad, const int dimension, const int* shape, const int jSize, const int dataSize, const int blockStride) {
    int numBlocks = dataSize / jSize;
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride);
        
        double* slice = new double[jSize];
        double* outputSlice = new double[jSize];
        
        for (int i = 0; i < jSize; i++) {
            int index = blockStartIndex + i * blockStride;
            slice[i] = x[index];
            outputSlice[i] = outputGrad[index];
        }
        
        double* jacobian = new double[jSize * jSize];
        
        for (int i = 0; i < jSize; i++) {
            for (int j = 0; j < jSize; j++) {
                jacobian[i * jSize + j] = (i == j) ? slice[i] * (1 - slice[i]) : -(slice[i] * slice[j]);
            }
        }
        int aShape[] = {jSize, jSize};
        int bShape[] = {jSize, 1};
        double* output = matrixMultiplyD(jacobian, outputSlice, aShape, bShape);
        for (int i = 0; i < jSize; i++) {
            int index = blockStartIndex + i * blockStride;
            y[index] = output[i];
        }
    }
}
/*
static void concatenate(const float** x, float* y, const int** shapes, const int* dataSizes, const int dimension, const int* lengthSizes, const int* blockStrides) {
    
    for ()
}*/

static void concatenate(const float** tensors, const int** shapes, const int num_tensors, const int num_dims, const int dimension, int* outputShape, float* y, const int* jSizes, const int inverseDimension) {
    int total = 1;
    for (int i = 0; i < num_dims; i++) {
        total *= outputShape[i];
    }
    float* result = new float[total];
    
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride);
        float* resultSlice = new float[inverseDimension];
        for (int tensor = 0; tensor < num_tensors; tensor++) {
            int blockLength = 0
            float* slice = new float[jSizes[tensor]];
            //float* outputSlice = new float[jSizes[tensor]];
                
            for (int i = 0; i < jSize; i++) {
                int index = blockStartIndex + i * blockStride;
                slice[i] = x[tensor][index];
                //outputSlice[i] = outputGrad[index];
            }
        }
            
        float* jacobian = new float[jSize * jSize];
            
        for (int i = 0; i < jSize; i++) {
            for (int j = 0; j < jSize; j++) {
                jacobian[i * jSize + j] = (i == j) ? slice[i] * (1 - slice[i]) : -(slice[i] * slice[j]);
            }
        }
        int aShape[] = {jSize, jSize};
        int bShape[] = {jSize, 1};
        float* output = matrixMultiply(jacobian, outputSlice, aShape, bShape);
        for (int i = 0; i < jSize; i++) {
            int index = blockStartIndex + i * blockStride;
            y[index] = output[i];
        }
    }
}
