#ifndef IMAGE_STRUCT_H
#define IMAGE_STRUCT_H

#include <vector>
#include <cmath>

// Image structure shared between proj.cpp and simd_filters.cpp
struct Image {
    int width;
    int height;
    std::vector<float> data;
    
    Image(int w, int h) : width(w), height(h), data(w * h, 0.0f) {}
    
    float& at(int x, int y) { return data[y * width + x]; }
    const float& at(int x, int y) const { return data[y * width + x]; }
};

// Utility functions (inline to avoid multiple definition errors)
inline int clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

inline std::vector<float> create_gaussian_kernel(int size, float sigma) {
    std::vector<float> kernel(size * size);
    int radius = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dx = x - radius;
            int dy = y - radius;
            float value = exp(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            kernel[y * size + x] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
    
    return kernel;
}

#endif // IMAGE_STRUCT_H

