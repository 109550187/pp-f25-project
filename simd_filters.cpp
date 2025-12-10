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
// Box filter - Sliding window optimization (scalar version for clarity)
Image box_filter_sliding_window(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
        // Left boundary: scalar processing for pixels where kernel extends beyond left edge
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
        
        // Calculate sum for first safe pixel (x = radius)
        if (radius < input.width - radius) {
            float current_sum = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, input.height - 1);
                const float* row_ptr = &input.data[py * input.width];
                for (int kx = -radius; kx <= radius; kx++) {
                    current_sum += row_ptr[radius + kx];
                }
            }
            output.at(radius, y) = current_sum * inv_count;
            
            // Middle region: use sliding window optimization
            for (x = radius + 1; x < input.width - radius; x++) {
                // Subtract left column and add right column
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
        }
        
        // Right boundary: scalar fallback
        for (x = std::max(radius + 1, input.width - radius); x < input.width; x++) {
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
// Box filter - SIMD + Sliding Window optimization (Advanced)
Image box_filter_simd_sliding_window(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    int width = input.width;
    int height = input.height;

    // We use OpenMP to parallelize rows. 
    // Since we use a temporary buffer, each thread needs its own memory.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        // 1. ALLOCATE BUFFER (Thread-Local)
        // Stores vertical sums: vert_sums[x] = sum(input[x, y-r]...input[x, y+r])
        // 4096 floats is 16KB, which fits easily in L1 Cache.
        std::vector<float> vert_sums(width); 

        // 2. VERTICAL PASS (SIMD Optimized)
        // We compute the vertical sums for the entire row first.
        // This effectively flattens the 2D kernel into a 1D line.
        int x = 0;
        
        // Main SIMD Loop for Vertical Sums
        int safe_simd_end = width - 8;
        for (; x <= safe_simd_end; x += 8) {
            __m256 vsum = _mm256_setzero_ps();
            
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, height - 1);
                // Load 8 pixels from the specific row
                __m256 pixel_vec = _mm256_loadu_ps(&input.data[py * width + x]);
                vsum = _mm256_add_ps(vsum, pixel_vec);
            }
            // Store into temp buffer
            _mm256_storeu_ps(&vert_sums[x], vsum);
        }

        // Scalar fallback for remaining right edge (Vertical)
        for (; x < width; x++) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                int py = clamp(y + ky, 0, height - 1);
                sum += input.data[py * width + x];
            }
            vert_sums[x] = sum;
        }

        // 3. HORIZONTAL PASS (Scalar Sliding Window)
        // Now we simply slide across 'vert_sums' in O(1) time.
        
        // Initialize the window for x=0
        float window_sum = 0.0f;
        for (int kx = -radius; kx <= radius; kx++) {
            int px = clamp(kx, 0, width - 1);
            window_sum += vert_sums[px];
        }
        output.at(0, y) = window_sum * inv_count;

        // Slide the window for the rest of the row
        // Formula: NewSum = OldSum - TrailingVal + LeadingVal
        for (int i = 1; i < width; i++) {
            // Remove the pixel that just slid out (Left)
            int remove_idx = clamp(i - radius - 1, 0, width - 1);
            // Add the pixel that just slid in (Right)
            int add_idx = clamp(i + radius, 0, width - 1);

            window_sum = window_sum - vert_sums[remove_idx] + vert_sums[add_idx];
            
            output.at(i, y) = window_sum * inv_count;
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

// ============================================================================
// TILED IMPLEMENTATIONS
// ============================================================================

// Define tile size - tune this based on cache size
// Typical L1 cache: 32KB, L2: 256KB, L3: several MB
// For float data (4 bytes), 64x64 tile = 16KB, fits in L1
constexpr int TILE_SIZE = 64;

// Box filter - SIMD + Tiled (basic version)
Image box_filter_simd_tiled(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int tile_y = 0; tile_y < input.height; tile_y += TILE_SIZE) {
        for (int tile_x = 0; tile_x < input.width; tile_x += TILE_SIZE) {
            int y_end = min(tile_y + TILE_SIZE, input.height);
            int x_end = min(tile_x + TILE_SIZE, input.width);
            
            // Process each output pixel within the tile
            for (int y = tile_y; y < y_end; y++) {
                for (int x = tile_x; x < x_end; x++) {
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
        }
    }
    
    return output;
}

// Box filter - SIMD optimized + Tiled (matches box_filter_simd_optimized structure)
Image box_filter_simd_optimized_tiled(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    __m256 inv_count_vec = _mm256_set1_ps(inv_count);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int tile_y = 0; tile_y < input.height; tile_y += TILE_SIZE) {
        for (int tile_x = 0; tile_x < input.width; tile_x += TILE_SIZE) {
            int y_end = min(tile_y + TILE_SIZE, input.height);
            int x_end = min(tile_x + TILE_SIZE, input.width);
            
            for (int y = tile_y; y < y_end; y++) {
                // Determine boundaries within this tile
                int tile_left_boundary = max(tile_x, 0);
                int tile_right_boundary = min(x_end, input.width);
                
                int x = tile_left_boundary;
                
                // Left boundary: scalar processing
                // Process pixels where x < radius OR x < tile_x + radius (tile edge)
                int left_end = min(radius, tile_right_boundary);
                for (; x < left_end; x++) {
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
                
                // Middle region: fully vectorized - process 8 output pixels at once
                // Safe region where we can load 8 consecutive pixels without boundary checks
                int safe_end = min(tile_right_boundary - 8, input.width - radius - 8);
                for (; x <= safe_end; x += 8) {
                    __m256 sum_vec = _mm256_setzero_ps();
                    
                    // Accumulate kernel for 8 output pixels simultaneously
                    for (int ky = -radius; ky <= radius; ky++) {
                        int py = clamp(y + ky, 0, input.height - 1);
                        const float* row_ptr = &input.data[py * input.width];
                        
                        // Sum all kernel rows for the 8 output pixels
                        for (int kx = -radius; kx <= radius; kx++) {
                            // Direct load: no boundary check needed in safe region
                            __m256 pixel_vec = _mm256_loadu_ps(&row_ptr[x + kx]);
                            sum_vec = _mm256_add_ps(sum_vec, pixel_vec);
                        }
                    }
                    
                    // Divide by count and store
                    __m256 result_vec = _mm256_mul_ps(sum_vec, inv_count_vec);
                    _mm256_storeu_ps(&output.data[y * input.width + x], result_vec);
                }
                
                // Right boundary: scalar processing
                for (; x < tile_right_boundary; x++) {
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
        }
    }
    
    return output;
}

// Gaussian filter - SIMD + Tiled
Image gaussian_filter_simd_tiled(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    std::vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int tile_y = 0; tile_y < input.height; tile_y += TILE_SIZE) {
        for (int tile_x = 0; tile_x < input.width; tile_x += TILE_SIZE) {
            int y_end = min(tile_y + TILE_SIZE, input.height);
            int x_end = min(tile_x + TILE_SIZE, input.width);
            
            for (int y = tile_y; y < y_end; y++) {
                int x = tile_x;
                
                // SIMD processing (8 pixels at a time)
                int simd_end = x_end - 8;
                for (; x <= simd_end; x += 8) {
                    __m256 sum_vec = _mm256_setzero_ps();
                    
                    for (int ky = -radius; ky <= radius; ky++) {
                        for (int kx = -radius; kx <= radius; kx++) {
                            int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
                            __m256 kernel_val = _mm256_set1_ps(kernel[kernel_idx]);
                            
                            float pixels[8];
                            for (int i = 0; i < 8; i++) {
                                int px = clamp(x + i + kx, 0, input.width - 1);
                                int py = clamp(y + ky, 0, input.height - 1);
                                pixels[i] = input.at(px, py);
                            }
                            __m256 pixel_vec = _mm256_loadu_ps(pixels);
                            
                            sum_vec = _mm256_fmadd_ps(pixel_vec, kernel_val, sum_vec);
                        }
                    }
                    
                    _mm256_storeu_ps(&output.data[y * input.width + x], sum_vec);
                }
                
                // Scalar fallback for remaining pixels
                for (; x < x_end; x++) {
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
        }
    }
    
    return output;
}

Image gaussian_filter_simd_optimized_tiled(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    std::vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    int width = input.width;
    int height = input.height;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int tile_y = 0; tile_y < height; tile_y += TILE_SIZE) {
        for (int tile_x = 0; tile_x < width; tile_x += TILE_SIZE) {
            
            int y_end = std::min(tile_y + TILE_SIZE, height);
            int x_end = std::min(tile_x + TILE_SIZE, width);

            for (int y = tile_y; y < y_end; y++) {
                int x = tile_x;

                if (tile_x < radius) {
                    int safe_simd_start = radius;
                    int scalar_limit = std::min(x_end, safe_simd_start);
                    
                    for (; x < scalar_limit; x++) {
                        float sum = 0.0f;
                        for (int ky = -radius; ky <= radius; ky++) {
                            int py = clamp(y + ky, 0, height - 1);
                            const float* row_ptr = &input.data[py * width];
                            for (int kx = -radius; kx <= radius; kx++) {
                                int px = clamp(x + kx, 0, width - 1);
                                int k_idx = (ky + radius) * kernel_size + (kx + radius);
                                sum += row_ptr[px] * kernel[k_idx];
                            }
                        }
                        output.at(x, y) = sum;
                    }
                }

                int vector_limit = width - radius - 8;
                int simd_end = std::min(x_end, vector_limit + 8);

                for (; x <= simd_end - 8; x += 8) {
                    __m256 sum_vec = _mm256_setzero_ps();

                    for (int ky = -radius; ky <= radius; ky++) {
                        int py = clamp(y + ky, 0, height - 1);
                        const float* row_ptr = &input.data[py * width];
                        
                        const float* k_row = &kernel[(ky + radius) * kernel_size + radius];

                        for (int kx = -radius; kx <= radius; kx++) {
                            __m256 k_val = _mm256_set1_ps(k_row[kx]);
                            __m256 p_val = _mm256_loadu_ps(&row_ptr[x + kx]);
                            sum_vec = _mm256_fmadd_ps(p_val, k_val, sum_vec);
                        }
                    }
                    _mm256_storeu_ps(&output.data[y * width + x], sum_vec);
                }

                for (; x < x_end; x++) {
                    float sum = 0.0f;
                    for (int ky = -radius; ky <= radius; ky++) {
                        int py = clamp(y + ky, 0, height - 1);
                        const float* row_ptr = &input.data[py * width];
                        for (int kx = -radius; kx <= radius; kx++) {
                            int px = clamp(x + kx, 0, width - 1);
                            int k_idx = (ky + radius) * kernel_size + (kx + radius);
                            sum += row_ptr[px] * kernel[k_idx];
                        }
                    }
                    output.at(x, y) = sum;
                }
            }
        }
    }
    return output;
}

// Box filter - Sliding window + Tiled
Image box_filter_sliding_window_tiled(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int tile_y = 0; tile_y < input.height; tile_y += TILE_SIZE) {
        for (int tile_x = 0; tile_x < input.width; tile_x += TILE_SIZE) {
            int y_end = std::min(tile_y + TILE_SIZE, input.height);
            int x_end = std::min(tile_x + TILE_SIZE, input.width);
            
            for (int y = tile_y; y < y_end; y++) {
                int x = tile_x;
                
                // Left boundary within tile (pixels where kernel extends beyond left edge)
                for (; x < std::min(tile_x + radius, x_end) && x < radius; x++) {
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
                
                // Start sliding window from first safe pixel in tile
                int x_start = std::max(x, radius);
                if (x_start < x_end && x_start < input.width - radius) {
                    // Calculate initial sum for first pixel
                    float current_sum = 0.0f;
                    for (int ky = -radius; ky <= radius; ky++) {
                        int py = clamp(y + ky, 0, input.height - 1);
                        const float* row_ptr = &input.data[py * input.width];
                        for (int kx = -radius; kx <= radius; kx++) {
                            current_sum += row_ptr[x_start + kx];
                        }
                    }
                    output.at(x_start, y) = current_sum * inv_count;
                    
                    // Sliding window for middle region
                    int safe_end = std::min(x_end, input.width - radius);
                    for (x = x_start + 1; x < safe_end; x++) {
                        for (int ky = -radius; ky <= radius; ky++) {
                            int py = clamp(y + ky, 0, input.height - 1);
                            const float* row_ptr = &input.data[py * input.width];
                            
                            int left_col = x - radius - 1;
                            current_sum -= row_ptr[left_col];
                            
                            int right_col = x + radius;
                            current_sum += row_ptr[right_col];
                        }
                        
                        output.at(x, y) = current_sum * inv_count;
                    }
                    
                    x = safe_end;
                }
                
                // Right boundary within tile
                for (; x < x_end; x++) {
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
        }
    }
    
    return output;
}

Image box_filter_simd_sliding_window_tiled(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    int width = input.width;
    int height = input.height;

    // Determine buffer size needed for vertical sums
    // We need the tile width + the "apron" (radius pixels on left and right)
    // so we can slide the window correctly at the tile edges.
    int max_buffer_size = TILE_SIZE + 2 * radius;

    #pragma omp parallel
    {
        // Thread-local buffer to store vertical sums. 
        // Allocating once per thread saves heap overhead.
        std::vector<float> v_buf(max_buffer_size);

        #pragma omp for collapse(2) schedule(static)
        for (int tile_y = 0; tile_y < height; tile_y += TILE_SIZE) {
            for (int tile_x = 0; tile_x < width; tile_x += TILE_SIZE) {
                
                int y_end = std::min(tile_y + TILE_SIZE, height);
                int x_end = std::min(tile_x + TILE_SIZE, width);
                int cur_tile_w = x_end - tile_x;

                // Define the range of X we need to calculate vertical sums for.
                // We need (radius) pixels to the left and right of the tile 
                // to support the horizontal sliding window.
                int buf_start_x = tile_x - radius;
                int buf_end_x   = x_end + radius;
                int buf_width   = buf_end_x - buf_start_x;

                for (int y = tile_y; y < y_end; y++) {
                    
                    // --- STEP 1: VERTICAL PASS (SIMD) ---
                    // Compute vertical sums for the range [buf_start_x, buf_end_x]
                    // and store them in v_buf.
                    
                    int bx = 0; // index inside v_buf
                    int gx = buf_start_x; // global x index
                    
                    // 1a. SIMD Vertical Sums
                    for (; bx <= buf_width - 8; bx += 8, gx += 8) {
                        __m256 vsum = _mm256_setzero_ps();
                        
                        for (int ky = -radius; ky <= radius; ky++) {
                            int py = clamp(y + ky, 0, height - 1);
                            
                            // Safe global X handling for clamp
                            // We load 8 pixels starting at 'gx'
                            // Since gx might be negative (left apron), we must clamp carefully.
                            // However, clamping vector loads is slow. 
                            // Optimization: Check if we are fully inside image bounds.
                            if (gx >= 0 && gx + 7 < width) {
                                __m256 p = _mm256_loadu_ps(&input.data[py * width + gx]);
                                vsum = _mm256_add_ps(vsum, p);
                            } else {
                                // Fallback for boundary overlap (rare)
                                float tmp[8];
                                for(int i=0; i<8; i++) 
                                    tmp[i] = input.data[py * width + clamp(gx+i, 0, width-1)];
                                __m256 p = _mm256_loadu_ps(tmp);
                                vsum = _mm256_add_ps(vsum, p);
                            }
                        }
                        _mm256_storeu_ps(&v_buf[bx], vsum);
                    }

                    // 1b. Scalar Cleanup for Vertical Sums
                    for (; bx < buf_width; bx++, gx++) {
                        float sum = 0.0f;
                        for (int ky = -radius; ky <= radius; ky++) {
                            int py = clamp(y + ky, 0, height - 1);
                            int px = clamp(gx, 0, width - 1); // Clamp X 
                            sum += input.data[py * width + px];
                        }
                        v_buf[bx] = sum;
                    }

                    // --- STEP 2: HORIZONTAL PASS (Sliding Window) ---
                    // Now slide across v_buf to generate final output
                    // v_buf[radius] corresponds to global pixel tile_x
                    
                    // Initialize window sum
                    // The window for the first pixel in the tile (tile_x) 
                    // covers v_buf[0] to v_buf[2*radius]
                    float window_sum = 0.0f;
                    for (int k = 0; k < 2 * radius + 1; k++) {
                        window_sum += v_buf[k];
                    }
                    output.at(tile_x, y) = window_sum * inv_count;

                    // Slide for the rest of the tile
                    // New = Old - Left + Right
                    for (int i = 1; i < cur_tile_w; i++) {
                        window_sum -= v_buf[i - 1];               // Remove trailing
                        window_sum += v_buf[i + 2 * radius];      // Add leading
                        output.at(tile_x + i, y) = window_sum * inv_count;
                    }
                }
            }
        }
    }
    return output;
}

// OpenMP + SIMD auto-vectorization + Tiled
Image box_filter_openmp_simd_auto_tiled(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    float inv_count = 1.0f / (kernel_size * kernel_size);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int tile_y = 0; tile_y < input.height; tile_y += TILE_SIZE) {
        for (int tile_x = 0; tile_x < input.width; tile_x += TILE_SIZE) {
            int y_end = min(tile_y + TILE_SIZE, input.height);
            int x_end = min(tile_x + TILE_SIZE, input.width);
            
            for (int y = tile_y; y < y_end; y++) {
                #pragma omp simd
                for (int x = tile_x; x < x_end; x++) {
                    float sum = 0.0f;
                    
                    for (int ky = -radius; ky <= radius; ky++) {
                        for (int kx = -radius; kx <= radius; kx++) {
                            int px = clamp(x + kx, 0, input.width - 1);
                            int py = clamp(y + ky, 0, input.height - 1);
                            sum += input.at(px, py);
                        }
                    }
                    
                    output.at(x, y) = sum * inv_count;
                }
            }
        }
    }
    
    return output;
}

Image gaussian_filter_openmp_simd_auto_tiled(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    std::vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int tile_y = 0; tile_y < input.height; tile_y += TILE_SIZE) {
        for (int tile_x = 0; tile_x < input.width; tile_x += TILE_SIZE) {
            int y_end = min(tile_y + TILE_SIZE, input.height);
            int x_end = min(tile_x + TILE_SIZE, input.width);
            
            for (int y = tile_y; y < y_end; y++) {
                #pragma omp simd
                for (int x = tile_x; x < x_end; x++) {
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
        }
    }
    
    return output;
}

