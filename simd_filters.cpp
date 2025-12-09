#include "image_struct.h"  // Image structure and utility functions
#include <vector>
#include <omp.h>
#include <immintrin.h>  // AVX2 intrinsics
#include <algorithm>

using namespace std;

// ============================================================================
// SIMD IMPLEMENTATIONS (AVX2)
// ============================================================================

// Box filter - SIMD version (manual vectorization)
// Note: Box filter is challenging to vectorize due to boundary handling
// This version accumulates all kernel pixels and then divides
Image box_filter_simd(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
        // Process each output pixel
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            
            // Accumulate all kernel pixels
            // For SIMD, we can vectorize when processing multiple kernel rows
            // But for correctness, we'll use a simpler scalar approach with SIMD hints
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                
                // Process kernel row - try to use SIMD for consecutive pixels when possible
                int kx = -radius;
                
                // Vectorize when we have enough consecutive pixels (away from boundaries)
                if (x >= radius && x + radius < input.width - 7) {
                    // Can safely load 8 consecutive pixels
                    for (; kx <= radius - 7; kx += 8) {
                        __m256 pixel_vec = _mm256_loadu_ps(&row_ptr[x + kx]);
                        // Horizontal sum
                        __m128 low = _mm256_extractf128_ps(pixel_vec, 0);
                        __m128 high = _mm256_extractf128_ps(pixel_vec, 1);
                        __m128 sum128 = _mm_add_ps(low, high);
                        sum128 = _mm_hadd_ps(sum128, sum128);
                        sum128 = _mm_hadd_ps(sum128, sum128);
                        sum += _mm_cvtss_f32(sum128);
                    }
                }
                
                // Handle remaining pixels (scalar)
                for (; kx <= radius; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    sum += row_ptr[px];
                }
            }
            
            output.at(x, y) = sum * inv_count;
        }
    }
    
    return output;
}

// Gaussian filter - SIMD version (manual vectorization)
Image gaussian_filter_simd(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    std::vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
        int x = 0;
        
        // Process 8 pixels at a time using AVX2
        for (; x <= input.width - 8; x += 8) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            // Apply weighted kernel
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
                    __m256 kernel_val = _mm256_set1_ps(kernel[kernel_idx]);
                    
                    // Load 8 pixels
                    float pixels[8];
                    for (int i = 0; i < 8; i++) {
                        int px = clamp(x + i + kx, 0, input.width - 1);
                        int py = clamp(y + ky, 0, input.height - 1);
                        pixels[i] = input.at(px, py);
                    }
                    __m256 pixel_vec = _mm256_loadu_ps(pixels);
                    
                    // Fused multiply-add: sum += pixel * kernel
                    sum_vec = _mm256_fmadd_ps(pixel_vec, kernel_val, sum_vec);
                }
            }
            
            // Store result
            _mm256_storeu_ps(&output.data[y * input.width + x], sum_vec);
        }
        
        // Handle remaining pixels (scalar fallback)
        for (; x < input.width; x++) {
            float sum = 0.0f;
            
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    int py = clamp(y + ky, 0, input.height - 1);
                    int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
                    sum += input.at(px, py) * kernel[kernel_idx];
                }
            }
            
            output.at(x, y) = sum;
        }
    }
    
    return output;
}

// Gaussian filter - SIMD optimized version (improved performance)
// Optimizations:
// 1. Separate boundary handling to avoid clamp() in hot path
// 2. Direct memory access instead of at() function calls
// 3. Better cache utilization with contiguous memory access
Image gaussian_filter_simd_optimized(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    std::vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
        // Left boundary: scalar processing
        int x = 0;
        for (; x < radius && x < input.width; x++) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
                    sum += row_ptr[px] * kernel[kernel_idx];
                }
            }
            output.at(x, y) = sum;
        }
        
        // Middle region: fully vectorized (safe to load 8 consecutive pixels)
        int safe_end = input.width - radius - 8;
        for (; x <= safe_end; x += 8) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            // Process kernel
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                
                for (int kx = -radius; kx <= radius; kx++) {
                    int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
                    __m256 kernel_val = _mm256_set1_ps(kernel[kernel_idx]);
                    
                    // Direct load: no boundary check needed in safe region
                    __m256 pixel_vec = _mm256_loadu_ps(&row_ptr[x + kx]);
                    
                    // FMA: sum += pixel * kernel (for all 8 output pixels)
                    sum_vec = _mm256_fmadd_ps(pixel_vec, kernel_val, sum_vec);
                }
            }
            
            // Store 8 results
            _mm256_storeu_ps(&output.data[y * input.width + x], sum_vec);
        }
        
        // Right boundary: scalar processing
        for (; x < input.width; x++) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
                    sum += row_ptr[px] * kernel[kernel_idx];
                }
            }
            output.at(x, y) = sum;
        }
    }
    
    return output;
}

// Box filter - SIMD optimized version
// Key optimization: process multiple output pixels simultaneously
Image box_filter_simd_optimized(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    __m256 inv_count_vec = _mm256_set1_ps(inv_count);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
        // Left boundary: scalar
        int x = 0;
        for (; x < radius && x < input.width; x++) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    sum += row_ptr[px];
                }
            }
            output.at(x, y) = sum * inv_count;
        }
        
        // Middle region: vectorized - process 8 output pixels at once
        int safe_end = input.width - radius - 8;
        for (; x <= safe_end; x += 8) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            // Accumulate kernel for 8 output pixels simultaneously
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                
                // Sum all kernel rows for the 8 output pixels
                for (int kx = -radius; kx <= radius; kx++) {
                    // Load 8 consecutive pixels (safe in middle region)
                    __m256 pixel_vec = _mm256_loadu_ps(&row_ptr[x + kx]);
                    sum_vec = _mm256_add_ps(sum_vec, pixel_vec);
                }
            }
            
            // Divide by count and store
            __m256 result_vec = _mm256_mul_ps(sum_vec, inv_count_vec);
            _mm256_storeu_ps(&output.data[y * input.width + x], result_vec);
        }
        
        // Right boundary: scalar
        for (; x < input.width; x++) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    sum += row_ptr[px];
                }
            }
            output.at(x, y) = sum * inv_count;
        }
    }
    
    return output;
}

// Box filter - OpenMP + SIMD (compiler auto-vectorization)
Image box_filter_openmp_simd_auto(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
        #pragma omp simd
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            int count = 0;
            
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    int py = clamp(y + ky, 0, input.height - 1);
                    sum += input.at(px, py);
                    count++;
                }
            }
            
            output.at(x, y) = sum * inv_count;
        }
    }
    
    return output;
}

// Gaussian filter - OpenMP + SIMD (compiler auto-vectorization)
Image gaussian_filter_openmp_simd_auto(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    std::vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
        #pragma omp simd
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    int py = clamp(y + ky, 0, input.height - 1);
                    int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
                    sum += input.at(px, py) * kernel[kernel_idx];
                }
            }
            
            output.at(x, y) = sum;
        }
    }
    
    return output;
}

