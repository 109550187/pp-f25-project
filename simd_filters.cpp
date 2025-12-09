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

// Box filter - Sliding window optimization (scalar version for clarity)
// Key insight: For box filter (uniform weights), adjacent output pixels share most kernel pixels
// Instead of recalculating O(kernel_size^2) pixels, we can:
// - Start with the sum for the first pixel
// - For each subsequent pixel: sum_new = sum_old - left_column + right_column
// This reduces complexity from O(kernel_size^2) to O(kernel_size) per pixel
Image box_filter_sliding_window(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
        // Left boundary: calculate first pixel normally
        int x = radius;
        if (x >= input.width) continue;
        
        // Calculate sum for first output pixel (x = radius)
        float current_sum = 0.0f;
        for (int ky = -radius; ky <= radius; ky++) {
            int py = clamp(y + ky, 0, input.height - 1);
            const float* row_ptr = &input.data[py * input.width];
            for (int kx = -radius; kx <= radius; kx++) {
                int px = clamp(x + kx, 0, input.width - 1);
                current_sum += row_ptr[px];
            }
        }
        output.at(x, y) = current_sum * inv_count;
        
        // Middle region: use sliding window optimization
        // For output pixel x, kernel covers columns [x-radius, x+radius]
        // For output pixel x+1, kernel covers columns [x-radius+1, x+radius+1]
        // So: sum(x+1) = sum(x) - sum of column(x-radius) + sum of column(x+radius+1)
        for (x = radius + 1; x < input.width - radius; x++) {
            // Subtract left column (x - radius - 1) and add right column (x + radius)
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                
                // Subtract: column that's leaving the kernel
                int left_col = x - radius - 1;
                current_sum -= row_ptr[left_col];
                
                // Add: column that's entering the kernel
                int right_col = x + radius;
                current_sum += row_ptr[right_col];
            }
            
            output.at(x, y) = current_sum * inv_count;
        }
        
        // Right boundary: scalar fallback
        for (x = input.width - radius; x < input.width; x++) {
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

// Box filter - SIMD + Sliding Window optimization (Advanced)
// Maintains 8 output pixel sums simultaneously and updates them using SIMD
// Key insight: For 8 output pixels [x, x+1, ..., x+7], each has its own sum
// When moving to next group [x+8, x+9, ..., x+15]:
//   - Each sum shifts: sum_new[i] = sum_old[i] - left_col[i] + right_col[i]
//   - We can use SIMD to update all 8 sums simultaneously
Image box_filter_simd_sliding_window(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    __m256 inv_count_vec = _mm256_set1_ps(inv_count);
    
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
                    sum += row_ptr[px];
                }
            }
            output.at(x, y) = sum * inv_count;
        }
        
        // Middle region: SIMD + Sliding Window with 8 simultaneous sums
        int safe_start = radius;
        int safe_end = input.width - radius - 8;
        
        if (safe_start < input.width && safe_end >= safe_start) {
            // Step 1: Calculate initial sums for first 8 output pixels (x = safe_start to safe_start+7)
            // Each output pixel i needs sum of kernel covering [safe_start+i-radius, safe_start+i+radius]
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                
                // For each column in kernel, accumulate across 8 output pixels
                for (int kx = -radius; kx <= radius; kx++) {
                    // Load 8 consecutive pixels starting at (safe_start + kx)
                    // These correspond to the same kernel offset for 8 different output pixels
                    int px = safe_start + kx;
                    if (px >= 0 && px + 7 < input.width) {
                        __m256 pixel_vec = _mm256_loadu_ps(&row_ptr[px]);
                        sum_vec = _mm256_add_ps(sum_vec, pixel_vec);
                    }
                }
            }
            
            // Store first 8 results
            __m256 result_vec = _mm256_mul_ps(sum_vec, inv_count_vec);
            _mm256_storeu_ps(&output.data[y * input.width + safe_start], result_vec);
            
            // Step 2: Use sliding window for remaining 8-pixel groups with SIMD updates
            // Strategy: For each group of 8 output pixels, compute column sums using SIMD
            // then update all 8 sums simultaneously
            for (x = safe_start + 8; x <= safe_end; x += 8) {
                // Extract the last sum from previous group (for output pixel x-1)
                float prev_sums[8];
                _mm256_storeu_ps(prev_sums, sum_vec);
                float base_sum = prev_sums[7]; // Sum for output pixel x-1
                
                // Compute column sums for all 8 output pixels using SIMD
                // For output pixel x+i: need to subtract column (x+i-radius-1) and add column (x+i+radius)
                // So we need columns: [x-radius-1, x-radius, ..., x-radius+6] to subtract
                //                  and [x+radius, x+radius+1, ..., x+radius+7] to add
                
                __m256 left_cols_delta = _mm256_setzero_ps();  // Sum of left columns to subtract
                __m256 right_cols_delta = _mm256_setzero_ps();  // Sum of right columns to add
                
                // Accumulate column sums across all kernel rows
                for (int ky = -radius; ky <= radius; ky++) {
                    int py = clamp(y + ky, 0, input.height - 1);
                    const float* row_ptr = &input.data[py * input.width];
                    
                    // Load 8 left columns to subtract: [x-radius-1, x-radius, ..., x-radius+6]
                    int left_start = x - radius - 1;
                    if (left_start >= 0 && left_start + 7 < input.width) {
                        __m256 left_vec = _mm256_loadu_ps(&row_ptr[left_start]);
                        left_cols_delta = _mm256_add_ps(left_cols_delta, left_vec);
                    }
                    
                    // Load 8 right columns to add: [x+radius, x+radius+1, ..., x+radius+7]
                    int right_start = x + radius;
                    if (right_start >= 0 && right_start + 7 < input.width) {
                        __m256 right_vec = _mm256_loadu_ps(&row_ptr[right_start]);
                        right_cols_delta = _mm256_add_ps(right_cols_delta, right_vec);
                    }
                }
                
                // Compute new sums: start from base_sum and accumulate deltas
                // For output pixel x+i: sum = base_sum + cumulative_delta[i]
                // where cumulative_delta[i] = sum of (right_col[j] - left_col[j]) for j=0 to i
                
                // Convert SIMD to array for easier manipulation
                float left_deltas[8], right_deltas[8];
                _mm256_storeu_ps(left_deltas, left_cols_delta);
                _mm256_storeu_ps(right_deltas, right_cols_delta);
                
                // Compute cumulative deltas and new sums
                float new_sums[8];
                float cumulative_delta = 0.0f;
                for (int i = 0; i < 8; i++) {
                    cumulative_delta += (right_deltas[i] - left_deltas[i]);
                    new_sums[i] = base_sum + cumulative_delta;
                }
                
                // Pack new sums into SIMD register
                sum_vec = _mm256_loadu_ps(new_sums);
                
                // Store results
                result_vec = _mm256_mul_ps(sum_vec, inv_count_vec);
                _mm256_storeu_ps(&output.data[y * input.width + x], result_vec);
            }
        }
        
        // Right boundary: scalar fallback
        for (x = std::max(safe_end + 8, radius); x < input.width; x++) {
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

