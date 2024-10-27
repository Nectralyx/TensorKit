//
//  Use this file to import your target's public headers that you would like to expose to Swift.
/*
 * Copyright (c) 2024 Nectralyx.
 * This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 */

#pragma once

#include <stdio.h>
#include <cstring>  // For std::memcpy
#include <iostream>
#include <algorithm> //For std::min
#include <cmath>
#include <numeric>
#include <vector>
#include <sys/sysctl.h>
#include <limits>
#include <arm_neon.h>
#include <chrono>
#import "../cxxLibrary.cpp"

// Define the block size for tiling
static int calculateBlockSize() {
    // L1 cache size per core (in bytes), this is an estimate
    const int cacheSize = 32 * 1024; // 32 KB typical L1 cache size
    const int cacheLineSize = 64;    // 64 bytes per cache line (common)

    // Estimate how many floats can fit in the cache (each float is 4 bytes)
    int blockSize = cacheSize / sizeof(float);

    // Align block size to cache line size for better memory access patterns
    blockSize = (blockSize / cacheLineSize) * cacheLineSize;
    std::cout << "Optimal Block Size Found: " << blockSize << "\n";
    return blockSize;
}

static int BLOCK_SIZE = calculateBlockSize();

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
    int aRows = aShape[0];
    int aCols = aShape[1];
    int bRows = bShape[0];
    int bCols = bShape[1];

    if (aCols != bRows) {
        std::cerr << "Matrix dimensions do not match for multiplication!" << std::endl;
        return nullptr;
    }

    float* result = new float[aRows * bCols](); // Initialize to zero

    // Perform matrix multiplication
    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < bCols; ++j) {
            float sum = 0.0f; // Use a temporary variable to accumulate the result
            for (int k = 0; k < aCols; ++k) {
                sum += a[i * aCols + k] * b[k * bCols + j];
            }
            result[i * bCols + j] = sum; // Store the result
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

float32_t sum_float16x8_to_float32(float16x8_t vec) {
    // Split float16x8_t into two float16x4_t
    float16x4_t low = vget_low_f16(vec);
    float16x4_t high = vget_high_f16(vec);

    // Sum both halves pairwise: result is two float16x4_t vectors
    float16x4_t sum1 = vadd_f16(low, high); // Add the lower 4 and upper 4 values

    // Pairwise add within each float16x4_t
    sum1 = vpadd_f16(sum1, sum1); // Now we have [x1 + x2, x3 + x4, x5 + x6, x7 + x8]

    // Final pairwise addition to reduce it to a single value
    float16x4_t sum2 = vpadd_f16(sum1, sum1); // Now [sum]

    // Convert the final sum (float16) to float32
    float32x4_t result_f32 = vcvt_f32_f16(sum2);

    // Return the first lane as float32 scalar
    return vgetq_lane_f32(result_f32, 0);
}

static void softmaxJacobian(const float* x, float* y, const float* outputGrad, const int dimension, const int* shape, const int jSize, const int dataSize, const int blockStride) {
    int numBlocks = dataSize / jSize;

    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride);

        for (int i = 0; i < jSize; i++) {
            int index = blockStartIndex + i * blockStride;
            float weighted_sum = 0.0f;

            // Initialize NEON vectors for summing partial products
            float16x8_t sum_vec1 = vdupq_n_f16(0.0f);  // First group of 4 elements
            float16x8_t sum_vec2 = vdupq_n_f16(0.0f);  // Second group of 4 elements
            float16x8_t sum_vec3 = vdupq_n_f16(0.0f);  // Third group of 4 elements
            float16x8_t sum_vec4 = vdupq_n_f16(0.0f);  // Fourth group of 4 elements

            // Compute the dot product for 32 elements at a time
            int j = 0;
            for (; j <= jSize - 32; j += 32) {
                int idx1 = blockStartIndex + j * blockStride;
                int idx2 = blockStartIndex + (j + 4) * blockStride;
                int idx3 = blockStartIndex + (j + 8) * blockStride;
                int idx4 = blockStartIndex + (j + 12) * blockStride;
                int idx5 = blockStartIndex + (j + 16) * blockStride;
                int idx6 = blockStartIndex + (j + 20) * blockStride;
                int idx7 = blockStartIndex + (j + 24) * blockStride;
                int idx8 = blockStartIndex + (j + 28) * blockStride;

                // Load 16 consecutive floats from x and outputGrad (split into four 8-element vectors)
                float32x4_t x_vec1 = vld1q_f32(&x[idx1]);
                float32x4_t grad_vec1 = vld1q_f32(&outputGrad[idx1]);
                float32x4_t x_vec2 = vld1q_f32(&x[idx2]);
                float32x4_t grad_vec2 = vld1q_f32(&outputGrad[idx2]);
                float32x4_t x_vec3 = vld1q_f32(&x[idx3]);
                float32x4_t grad_vec3 = vld1q_f32(&outputGrad[idx3]);
                float32x4_t x_vec4 = vld1q_f32(&x[idx4]);
                float32x4_t grad_vec4 = vld1q_f32(&outputGrad[idx4]);
                float32x4_t x_vec5 = vld1q_f32(&x[idx5]);
                float32x4_t grad_vec5 = vld1q_f32(&outputGrad[idx5]);
                float32x4_t x_vec6 = vld1q_f32(&x[idx6]);
                float32x4_t grad_vec6 = vld1q_f32(&outputGrad[idx6]);
                float32x4_t x_vec7 = vld1q_f32(&x[idx7]);
                float32x4_t grad_vec7 = vld1q_f32(&outputGrad[idx7]);
                float32x4_t x_vec8 = vld1q_f32(&x[idx8]);
                float32x4_t grad_vec8 = vld1q_f32(&outputGrad[idx8]);
                
                float16x4_t x_11 = vcvt_f16_f32(x_vec1);
                float16x4_t grad_11 = vcvt_f16_f32(grad_vec1);
                float16x4_t x_12 = vcvt_f16_f32(x_vec2);
                float16x4_t grad_12 = vcvt_f16_f32(grad_vec2);
                float16x4_t x_21 = vcvt_f16_f32(x_vec3);
                float16x4_t grad_21 = vcvt_f16_f32(grad_vec3);
                float16x4_t x_22 = vcvt_f16_f32(x_vec4);
                float16x4_t grad_22 = vcvt_f16_f32(grad_vec4);
                float16x4_t x_31 = vcvt_f16_f32(x_vec5);
                float16x4_t grad_31 = vcvt_f16_f32(grad_vec5);
                float16x4_t x_32 = vcvt_f16_f32(x_vec6);
                float16x4_t grad_32 = vcvt_f16_f32(grad_vec6);
                float16x4_t x_41 = vcvt_f16_f32(x_vec7);
                float16x4_t grad_41 = vcvt_f16_f32(grad_vec7);
                float16x4_t x_42 = vcvt_f16_f32(x_vec8);
                float16x4_t grad_42 = vcvt_f16_f32(grad_vec8);
                
                float16x8_t a_vec1 = vcombine_f16(x_11, x_12);
                float16x8_t grade_vec1 = vcombine_f16(grad_11, grad_22);
                float16x8_t a_vec2 = vcombine_f16(x_21, x_22);
                float16x8_t grade_vec2 = vcombine_f16(grad_21, grad_22);
                float16x8_t a_vec3 = vcombine_f16(x_31, x_32);
                float16x8_t grade_vec3 = vcombine_f16(grad_31, grad_32);
                float16x8_t a_vec4 = vcombine_f16(x_41, x_42);
                float16x8_t grade_vec4 = vcombine_f16(grad_41, grad_42);

                // Multiply x and outputGrad element-wise for both sets
                sum_vec1 = vfmaq_f32(sum_vec1, a_vec1, grade_vec1);
                sum_vec2 = vfmaq_f32(sum_vec2, a_vec2, grade_vec2);
                sum_vec3 = vfmaq_f32(sum_vec3, a_vec3, grade_vec3);
                sum_vec4 = vfmaq_f32(sum_vec4, a_vec4, grade_vec4);
            }

            // Horizontal add for both vectors
            weighted_sum = sum_float16x8_to_float32(sum_vec1) + sum_float16x8_to_float32(sum_vec2) + sum_float16x8_to_float32(sum_vec3) + sum_float16x8_to_float32(sum_vec4);  // Reduction of all lanes

            // Handle remaining elements (if jSize isn't divisible by 16)
            for (; j < jSize; j++) {
                int idx = blockStartIndex + j * blockStride;
                weighted_sum += x[idx] * outputGrad[idx];
            }

            // Compute the gradient update
            float grad_update = outputGrad[index] - weighted_sum;
            y[index] = x[index] * grad_update;
        }
    }
}

static void softmaxJacobianD(const double* x, double* y, const double* outputGrad, const int dimension, const int* shape, const int jSize, const int dataSize, const int blockStride) {
    int numBlocks = dataSize / jSize;
    
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        int blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride);

        for (int i = 0; i < jSize; i++) {
            int index = blockStartIndex + i * blockStride;
            double weighted_sum = 0.0;

            // Initialize NEON vectors for summing partial products
            float64x2_t sum_vec1 = vdupq_n_f64(0.0);  // First group of 2 elements
            float64x2_t sum_vec2 = vdupq_n_f64(0.0);  // Second group of 2 elements
            float64x2_t sum_vec3 = vdupq_n_f64(0.0);  // Third group of 2 elements
            float64x2_t sum_vec4 = vdupq_n_f64(0.0);  // Fourth group of 2 elements

            // Compute the dot product for 8 elements at a time
            int j = 0;
            for (; j <= jSize - 8; j += 8) {
                int idx1 = blockStartIndex + j * blockStride;
                int idx2 = blockStartIndex + (j + 2) * blockStride;
                int idx3 = blockStartIndex + (j + 4) * blockStride;
                int idx4 = blockStartIndex + (j + 6) * blockStride;

                // Load 8 consecutive floats from x and outputGrad (split into four 2-element vectors)
                float64x2_t x_vec1 = vld1q_f64(&x[idx1]);
                float64x2_t grad_vec1 = vld1q_f64(&outputGrad[idx1]);
                float64x2_t x_vec2 = vld1q_f64(&x[idx2]);
                float64x2_t grad_vec2 = vld1q_f64(&outputGrad[idx2]);
                float64x2_t x_vec3 = vld1q_f64(&x[idx3]);
                float64x2_t grad_vec3 = vld1q_f64(&outputGrad[idx3]);
                float64x2_t x_vec4 = vld1q_f64(&x[idx4]);
                float64x2_t grad_vec4 = vld1q_f64(&outputGrad[idx4]);

                // Multiply x and outputGrad element-wise for both sets
                sum_vec1 = vfmaq_f64(sum_vec1, x_vec1, grad_vec1);
                sum_vec2 = vfmaq_f64(sum_vec2, x_vec2, grad_vec2);
                sum_vec3 = vfmaq_f64(sum_vec3, x_vec3, grad_vec3);
                sum_vec4 = vfmaq_f64(sum_vec4, x_vec4, grad_vec4);
            }

            // Horizontal add for both vectors
            weighted_sum = vaddvq_f64(sum_vec1) + vaddvq_f64(sum_vec2) + vaddvq_f64(sum_vec3) + vaddvq_f64(sum_vec4);  // Reduction of all lanes

            // Handle remaining elements (if jSize isn't divisible by 8)
            for (; j < jSize; j++) {
                int idx = blockStartIndex + j * blockStride;
                weighted_sum += x[idx] * outputGrad[idx];
            }

            // Compute the gradient update
            float grad_update = outputGrad[index] - weighted_sum;
            y[index] = x[index] * grad_update;
        }
    }
}

static void concatenate(const float** x, float* y, const int numBlocks, const int blockStride, const int totalLength, const int* jSizes, const int xCount) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            std::cout << "Value Found: " << x[i][j] << "\n";
        }
    }
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        float* slice = new float[totalLength];
        int sliceIndex = 0;
        for (int tensor = 0; tensor < xCount; tensor++) {
            std::cout << "Tensor: " << tensor << "JSize: " << jSizes[0];
            const int jSize = jSizes[tensor];
            std::cout << "Tensor: " << tensor << "Pass 1";
            const int blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride);
            std::cout << "Tensor: " << tensor << "Pass 2";
            for (int i = 0; i < jSize; i++) {
                const int index = blockStartIndex + i * blockStride;
                slice[sliceIndex] = x[tensor][index];
                std::cout << "slice[" << sliceIndex << "] = x[" << tensor << "][" << index << "] = " << slice[sliceIndex] << "\n";
                std::cout << x[tensor][index] << "\n";
                sliceIndex += 1;
            }
            std::cout << "Tensor: " << tensor << "Pass 3";
        }
        
        const int jSize = totalLength;
        const int blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride);
        for (int i = 0; i < jSize; i++) {
            const int index = blockStartIndex + i * blockStride;
            y[index] = slice[i];
            std::cout << slice[i];
        }
    }
}

static void concatenateD(const double** x, double* y, const int numBlocks, const int blockStride, const int totalLength, const int* jSizes, const int xCount) {
    for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
        double* slice = new double[totalLength];
        int sliceIndex = 0;
        for (int tensor = 0; tensor < xCount; tensor++) {
            const int jSize = jSizes[tensor];
            const int blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride);
            for (int i = 0; i < jSize; i++) {
                const int index = blockStartIndex + i * blockStride;
                slice[sliceIndex] = x[tensor][index];
                sliceIndex += 1;
            }
        }
        const int jSize = totalLength;
        const int blockStartIndex = (blockIndex / blockStride) * (blockStride * jSize) + (blockIndex % blockStride);
        for (int i = 0; i < jSize; i++) {
            const int index = blockStartIndex + i * blockStride;
            y[index] = slice[i];
        }
    }
}

int* unravelIndex(const int index, const int* shape, const int shapeCount, const int* strides) {
    int* result = new int[shapeCount];
    int remainingIndex = index;
    for (int i = 0; i < shapeCount; i++) {
        result[i] = remainingIndex / strides[i];
        remainingIndex = remainingIndex % strides[i];
    }
    return result;
}

int ravelIndex(const int* index, const int* strides, const int shapeCount) {
    int flatIndex = 0;
    for (int i = 0; i < shapeCount; i++) {
        flatIndex += strides[i] * index[i];
    }
    return flatIndex;
}

static void permute(const float* data, const float* grad, float* result, float* gresult,
                    const int* shape, const int* order, const int* oldStrides,
                    const int* newStrides, const int dataSize, const int shapeCount) {
    
    // Divide dataSize into blocks
    #pragma omp parallel for
    for (int blockStart = 0; blockStart < dataSize; blockStart += BLOCK_SIZE) {
        // Process a block of size BLOCK_SIZE, or the remainder at the end
        int blockEnd = std::min(blockStart + BLOCK_SIZE, dataSize);
        
        for (int i = blockStart; i < blockEnd; i++) {
            const int* originalIndex = unravelIndex(i, shape, shapeCount, oldStrides);

            // Preallocate newIndex once outside inner loop
            int newIndex[shapeCount];  // Use stack allocation for speed

            // Compute newIndex using the permutation order
            for (int j = 0; j < shapeCount; j++) {
                newIndex[j] = originalIndex[order[j]];
            }

            // Compute the flattened index for the result array
            const int newFlatIndex = ravelIndex(newIndex, newStrides, shapeCount);

            // Assign values to the result arrays
            result[newFlatIndex] = data[i];
            gresult[newFlatIndex] = grad[i];
        }
    }
}

static void permuteD(const double* data, const double* grad, double* result, double* gresult, const int* shape, const int* order, const int* oldStrides, const int* newStrides, const int dataSize, const int shapeCount) {
    // Divide dataSize into blocks
    #pragma omp parallel for
    for (int blockStart = 0; blockStart < dataSize; blockStart += BLOCK_SIZE) {
        // Process a block of size BLOCK_SIZE, or the remainder at the end
        int blockEnd = std::min(blockStart + BLOCK_SIZE, dataSize);
        
        for (int i = blockStart; i < blockEnd; i++) {
            const int* originalIndex = unravelIndex(i, shape, shapeCount, oldStrides);

            // Preallocate newIndex once outside inner loop
            int newIndex[shapeCount];  // Use stack allocation for speed

            // Compute newIndex using the permutation order
            for (int j = 0; j < shapeCount; j++) {
                newIndex[j] = originalIndex[order[j]];
            }

            // Compute the flattened index for the result array
            const int newFlatIndex = ravelIndex(newIndex, newStrides, shapeCount);

            // Assign values to the result arrays
            result[newFlatIndex] = data[i];
            gresult[newFlatIndex] = grad[i];
        }
    }
}

static void permuteNoGrad(const float* data, float* result, const int* shape, const int* order, const int* oldStrides, const int* newStrides, const int dataSize, const int shapeCount) {
    // Divide dataSize into blocks
    #pragma omp parallel for schedule(dynamic)
    for (int blockStart = 0; blockStart < dataSize; blockStart += BLOCK_SIZE) {
        // Process a block of size BLOCK_SIZE, or the remainder at the end
        int blockEnd = std::min(blockStart + BLOCK_SIZE, dataSize);
        
        for (int i = blockStart; i < blockEnd; i++) {
            const int* originalIndex = unravelIndex(i, shape, shapeCount, oldStrides);

            // Preallocate newIndex once outside inner loop
            int newIndex[shapeCount];  // Use stack allocation for speed

            // Compute newIndex using the permutation order
            for (int j = 0; j < shapeCount; j++) {
                newIndex[j] = originalIndex[order[j]];
            }

            // Compute the flattened index for the result array
            const int newFlatIndex = ravelIndex(newIndex, newStrides, shapeCount);

            // Assign values to the result arrays
            result[newFlatIndex] = data[i];
        }
    }
}

static void permuteNoGradD(const double* data, double* result, const int* shape, const int* order, const int* oldStrides, const int* newStrides, const int dataSize, const int shapeCount) {
    // Divide dataSize into blocks
    #pragma omp parallel for
    for (int blockStart = 0; blockStart < dataSize; blockStart += BLOCK_SIZE) {
        // Process a block of size BLOCK_SIZE, or the remainder at the end
        int blockEnd = std::min(blockStart + BLOCK_SIZE, dataSize);
        
        for (int i = blockStart; i < blockEnd; i++) {
            const int* originalIndex = unravelIndex(i, shape, shapeCount, oldStrides);

            // Preallocate newIndex once outside inner loop
            int newIndex[shapeCount];  // Use stack allocation for speed

            // Compute newIndex using the permutation order
            for (int j = 0; j < shapeCount; j++) {
                newIndex[j] = originalIndex[order[j]];
            }

            // Compute the flattened index for the result array
            const int newFlatIndex = ravelIndex(newIndex, newStrides, shapeCount);

            // Assign values to the result arrays
            result[newFlatIndex] = data[i];
        }
    }
}

static void expand_along_dimension(const float* input, float* output, const int* input_shape, const int* target_shape, const int ndim, const int expand_dim) {
    // Calculate input size
    int input_size = 1;
    for (int i = 0; i < ndim; ++i) {
        input_size *= input_shape[i];
    }

    // Calculate strides for input and output
    int input_stride = 1;
    int target_stride = 1;
    for (int i = ndim - 1; i > expand_dim; --i) {
        input_stride *= input_shape[i];
        target_stride *= target_shape[i];
    }
    int repeats = target_shape[expand_dim] / input_shape[expand_dim];
    
    // Loop over input elements and copy them to the output multiple times
    for (int i = 0; i < input_size; i += input_stride) {
        // Copy the chunk input_stride times
        for (int r = 0; r < repeats; ++r) {
            std::memcpy(output + (i * repeats) + (r * target_stride), input + i, input_stride * sizeof(float));
        }
    }
}

static void expand_along_dimensionD(const double* input, double* output, const int* input_shape, const int* target_shape, const int ndim, const int expand_dim) {
    // Calculate input size
    int input_size = 1;
    for (int i = 0; i < ndim; ++i) {
        input_size *= input_shape[i];
    }

    // Calculate strides for input and output
    int input_stride = 1;
    int target_stride = 1;
    for (int i = ndim - 1; i > expand_dim; --i) {
        input_stride *= input_shape[i];
        target_stride *= target_shape[i];
    }
    int repeats = target_shape[expand_dim] / input_shape[expand_dim];
    
    // Loop over input elements and copy them to the output multiple times
    for (int i = 0; i < input_size; i += input_stride) {
        // Copy the chunk input_stride times
        for (int r = 0; r < repeats; ++r) {
            std::memcpy(output + (i * repeats) + (r * target_stride), input + i, input_stride * sizeof(float));
        }
    }
}

