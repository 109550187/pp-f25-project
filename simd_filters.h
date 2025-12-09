#ifndef SIMD_FILTERS_H
#define SIMD_FILTERS_H

// Forward declaration
struct Image;

// SIMD filter function declarations
Image box_filter_simd(const Image& input, int kernel_size);
Image gaussian_filter_simd(const Image& input, int kernel_size, float sigma);
Image box_filter_simd_optimized(const Image& input, int kernel_size);
Image gaussian_filter_simd_optimized(const Image& input, int kernel_size, float sigma);
Image box_filter_openmp_simd_auto(const Image& input, int kernel_size);
Image gaussian_filter_openmp_simd_auto(const Image& input, int kernel_size, float sigma);
Image box_filter_sliding_window(const Image& input, int kernel_size);
Image box_filter_simd_sliding_window(const Image& input, int kernel_size);

#endif // SIMD_FILTERS_H

