//
//  Basic Operation Functions.hpp
//  Synapse
//
//  Created by Morgan Keay on 2024-09-14.
//
/*
#ifndef Basic_Operation_Functions_hpp
#define Basic_Operation_Functions_hpp

#include <stdio.h>

template <typename T>
void addArrays(const T* a, const T* b, T* result, int size);

template <typename T>
void subtractArrays(const T* a, const T* b, T* result, int size);

template <typename T>
void multiplyArrays(const T* a, const T* b, T* result, int size);

template <typename T>
void divideArrays(const T* a, const T* b, T* result, int size);

#endif // Basic_Operation_Functions_hpp */


#ifndef Basic_Operation_Functions_hpp
#define Basic_Operation_Functions_hpp

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

#endif /* Basic_Operation_Functions_hpp */

