#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <omp.h>

using namespace std;

// ============================================================================
// IMAGE STRUCTURE
// ============================================================================
struct Image {
    int width;
    int height;
    vector<float> data;
    
    Image(int w, int h) : width(w), height(h), data(w * h, 0.0f) {}
    
    float& at(int x, int y) { return data[y * width + x]; }
    const float& at(int x, int y) const { return data[y * width + x]; }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Create a simple test image
Image create_test_image(int width, int height) {
    Image img(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img.at(x, y) = (float)(x + y) / (width + height);
        }
    }
    return img;
}

// Generate Gaussian kernel
vector<float> create_gaussian_kernel(int size, float sigma) {
    vector<float> kernel(size * size);
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

// Clamp value to range
inline int clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Measure execution time
template<typename Func>
double measure_time(Func func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    return diff.count();
}

// ============================================================================
// SERIAL IMPLEMENTATIONS (Baseline)
// ============================================================================

// Box filter - Serial version
Image box_filter_serial(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            int count = 0;
            
            // Apply kernel
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    int py = clamp(y + ky, 0, input.height - 1);
                    sum += input.at(px, py);
                    count++;
                }
            }
            
            output.at(x, y) = sum / count;
        }
    }
    
    return output;
}

// Gaussian filter - Serial version
Image gaussian_filter_serial(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            
            // Apply weighted kernel
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
// OPENMP IMPLEMENTATIONS
// ============================================================================

// Box filter - OpenMP Static
Image box_filter_openmp_static(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
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
            
            output.at(x, y) = sum / count;
        }
    }
    
    return output;
}

// Box filter - OpenMP Dynamic
Image box_filter_openmp_dynamic(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < input.height; y++) {
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
            
            output.at(x, y) = sum / count;
        }
    }
    
    return output;
}

// Box filter - OpenMP Guided
Image box_filter_openmp_guided(const Image& input, int kernel_size) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    
    #pragma omp parallel for schedule(guided)
    for (int y = 0; y < input.height; y++) {
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
            
            output.at(x, y) = sum / count;
        }
    }
    
    return output;
}

// Gaussian filter - OpenMP Static
Image gaussian_filter_openmp_static(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; y++) {
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

// Gaussian filter - OpenMP Dynamic
Image gaussian_filter_openmp_dynamic(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < input.height; y++) {
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

// Gaussian filter - OpenMP Guided
Image gaussian_filter_openmp_guided(const Image& input, int kernel_size, float sigma) {
    Image output(input.width, input.height);
    int radius = kernel_size / 2;
    vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);
    
    #pragma omp parallel for schedule(guided)
    for (int y = 0; y < input.height; y++) {
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
// MAIN PROGRAM
// ============================================================================

int main(int argc, char* argv[]) {
    // Configuration
    int width = 4096;
    int height = 4096;
    int kernel_size = 7;
    float sigma = 2.0f;

    // Set default number of threads to 1
    int num_threads = 1;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads < 1) {
            cout << "Invalid thread count. Using 1 thread." << endl;
            num_threads = 1;
        }
    }
    omp_set_num_threads(num_threads);
    
    cout << "=== Parallel Image Filtering - Baseline ===" << endl;
    cout << "Image: " << width << " x " << height << endl;
    cout << "Kernel: " << kernel_size << " x " << kernel_size << endl;
    cout << "Threads: " << omp_get_max_threads() << endl;

    // Create test image
    Image input = create_test_image(width, height);
    
    // ========================================================================
    // SERIAL BASELINE
    // ========================================================================
    cout << "=== SERIAL BASELINE ===" << endl;
    
    double serial_box_time = measure_time([&]() {
        Image result = box_filter_serial(input, kernel_size);
    });
    cout << "Box Filter (Serial):      " << serial_box_time << " s" << endl;
    
    double serial_gauss_time = measure_time([&]() {
        Image result = gaussian_filter_serial(input, kernel_size, sigma);
    });
    cout << "Gaussian Filter (Serial): " << serial_gauss_time << " s" << endl;
    
    // ========================================================================
    // OPENMP IMPLEMENTATIONS
    // ========================================================================
    cout << "\n=== OPENMP - STATIC SCHEDULING ===" << endl;
    double box_static_time = measure_time([&]() {
        Image result = box_filter_openmp_static(input, kernel_size);
    });
    cout << "Box Filter (OpenMP Static):      " << box_static_time << " s  (speedup: " 
         << serial_box_time/box_static_time << "x)" << endl;
    
    double gauss_static_time = measure_time([&]() {
        Image result = gaussian_filter_openmp_static(input, kernel_size, sigma);
    });
    cout << "Gaussian Filter (OpenMP Static): " << gauss_static_time << " s  (speedup: " 
         << serial_gauss_time/gauss_static_time << "x)" << endl;
    
    cout << "\n=== OPENMP - DYNAMIC SCHEDULING ===" << endl;
    double box_dynamic_time = measure_time([&]() {
        Image result = box_filter_openmp_dynamic(input, kernel_size);
    });
    cout << "Box Filter (OpenMP Dynamic):      " << box_dynamic_time << " s  (speedup: " 
         << serial_box_time/box_dynamic_time << "x)" << endl;
    
    double gauss_dynamic_time = measure_time([&]() {
        Image result = gaussian_filter_openmp_dynamic(input, kernel_size, sigma);
    });
    cout << "Gaussian Filter (OpenMP Dynamic): " << gauss_dynamic_time << " s  (speedup: " 
         << serial_gauss_time/gauss_dynamic_time << "x)" << endl;
    
    cout << "\n=== OPENMP - GUIDED SCHEDULING ===" << endl;
    double box_guided_time = measure_time([&]() {
        Image result = box_filter_openmp_guided(input, kernel_size);
    });
    cout << "Box Filter (OpenMP Guided):      " << box_guided_time << " s  (speedup: " 
         << serial_box_time/box_guided_time << "x)" << endl;
    
    double gauss_guided_time = measure_time([&]() {
        Image result = gaussian_filter_openmp_guided(input, kernel_size, sigma);
    });
    cout << "Gaussian Filter (OpenMP Guided): " << gauss_guided_time << " s  (speedup: " 
         << serial_gauss_time/gauss_guided_time << "x)" << endl;
    
    cout << "\nNote: Run with 'export OMP_NUM_THREADS=X' to change thread count" << endl;
    
    return 0;
}
