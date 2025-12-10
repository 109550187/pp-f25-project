#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string> 
#include <cmath> 
#include "simd_filters.h"

#define PROFILER_MODE

#include "proj.cpp"

using namespace std;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void prevent_dce(const Image& res) {
    volatile float d = res.data[0];
    (void)d;
}

// Compare two images and return a status string ("PASS" or "FAIL")
string verify_result(const Image& ref, const Image& test) {
    if (ref.width != test.width || ref.height != test.height) {
        return "SIZE_ERR";
    }
    
    const float epsilon = 1e-3f; 

    for (size_t i = 0; i < ref.data.size(); ++i) {
        if (std::abs(ref.data[i] - test.data[i]) > epsilon) {
            return "FAIL"; 
        }
    }
    return "PASS";
}

// Helper to print a table row
void print_row(string name, double t_box, double s_box, string v_box, double t_gauss, double s_gauss, string v_gauss) {
    cout << left << setw(35) << name 
         << fixed << setprecision(4) 
         << setw(9) << t_box << "s " 
         << setw(9) << "(" + to_string(s_box).substr(0,5) + "x) "
         << setw(6) << v_box << " |    "
         << setw(9) << t_gauss << "s "
         << setw(9) << "(" + to_string(s_gauss).substr(0,5) + "x) " 
         << setw(6) << v_gauss
         << endl;
}

// ============================================================================
// MAIN PROFILER
// ============================================================================

int main(int argc, char* argv[]) {
    // Mode: 0=All(Benchmark), 1=Serial, 2=OpenMP(Stage2), 3=Tiled(Stage3)
    int mode = 0;
    if (argc > 1) mode = atoi(argv[1]);

    int num_threads = omp_get_max_threads(); 
    if (argc > 2) {
        num_threads = atoi(argv[2]);
        if (num_threads < 1) num_threads = 1;
        omp_set_num_threads(num_threads);
    }

    // Setup Data
    int width = 4096;
    int height = 4096;
    int kernel_size = 7;
    float sigma = 2.0f;

    Image input = create_test_image(width, height);

    // Store Reference Images (Keep these alive)
    Image ref_box(width, height);
    Image ref_gauss(width, height);

    static double serial_box_time = 0.0;
    static double serial_gauss_time = 0.0;

    if (mode == 0) {
        cout << "Running Performance Benchmark (4096 x 4096)..." << endl;
        cout << "Active Threads: " << omp_get_max_threads() << endl; 
        cout << "---------------------------------------------------------------------------------------------------------------" << endl;
        cout << left << setw(35) << "Version" 
             << setw(30) << "Box Filter (Time/Spd/Chk)" 
             << " |    " 
             << setw(30) << "Gaussian Filter (Time/Spd/Chk)" << endl;
        cout << "---------------------------------------------------------------------------------------------------------------" << endl;

        // 1. SERIAL (BASELINE)
        {
            serial_box_time = measure_time([&]() { 
                ref_box = box_filter_serial(input, kernel_size); 
                prevent_dce(ref_box);
            });
            serial_gauss_time = measure_time([&]() { 
                ref_gauss = gaussian_filter_serial(input, kernel_size, sigma); 
                prevent_dce(ref_gauss);
            });
            print_row("Serial (Base)", serial_box_time, 1.0, "REF", serial_gauss_time, 1.0, "REF");
        }

        // 2. OPENMP - STATIC
        {
            Image test_box(width, height);
            Image test_gauss(width, height);
            
            // Warmup
            box_filter_openmp_static(input, kernel_size);
            
            double t_box = measure_time([&]() { 
                test_box = box_filter_openmp_static(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);

            // Warmup
            gaussian_filter_openmp_static(input, kernel_size, sigma);

            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_openmp_static(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);

            print_row("OpenMP (Static)", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        } // <--- Memory for test_box/test_gauss is FREED here

        // 3. OPENMP - DYNAMIC
        {
            Image test_box(width, height);
            Image test_gauss(width, height);

            box_filter_openmp_dynamic(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_openmp_dynamic(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);

            gaussian_filter_openmp_dynamic(input, kernel_size, sigma);
            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_openmp_dynamic(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);

            print_row("OpenMP (Dynamic)", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        }

        // 4. OPENMP - GUIDED
        {
            Image test_box(width, height);
            Image test_gauss(width, height);

            box_filter_openmp_guided(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_openmp_guided(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);

            gaussian_filter_openmp_guided(input, kernel_size, sigma);
            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_openmp_guided(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);

            print_row("OpenMP (Guided)", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        }

        // 5. TILED (Stage 3)
        {
            Image test_box(width, height);
            Image test_gauss(width, height);

            box_filter_tiled(input, kernel_size); 
            double t_box = measure_time([&]() { 
                test_box = box_filter_tiled(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);

            gaussian_filter_tiled(input, kernel_size, sigma);
            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_tiled(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);

            print_row("Tiled (Stage 3)", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        }
        
        cout << "---------------------------------------------------------------------------------------------------------------" << endl;
        cout << "SIMD IMPLEMENTATIONS" << endl;
        cout << "---------------------------------------------------------------------------------------------------------------" << endl;

        // 6. SIMD MANUAL
        {
            Image test_box(width, height);
            Image test_gauss(width, height);

            box_filter_simd(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_simd(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);
            
            gaussian_filter_simd(input, kernel_size, sigma);
            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_simd(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);
            
            print_row("SIMD", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        }

        // 7. SIMD OPTIMIZED
        {
            Image test_box(width, height);
            Image test_gauss(width, height);

            box_filter_simd_optimized(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_simd_optimized(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);
            
            gaussian_filter_simd_optimized(input, kernel_size, sigma);
            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_simd_optimized(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);
            
            print_row("SIMD Optimized", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        }

        // 8. OPENMP SIMD AUTO
        {
            Image test_box(width, height);
            Image test_gauss(width, height);

            box_filter_openmp_simd_auto(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_openmp_simd_auto(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);

            gaussian_filter_openmp_simd_auto(input, kernel_size, sigma);
            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_openmp_simd_auto(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);
            
            print_row("OpenMP SIMD Auto", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        }
        
        // 9. BOX FILTER - Sliding Window
        {
            Image test_box(width, height);

            box_filter_sliding_window(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_sliding_window(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);
            print_row("Box - Sliding Window", t_box, serial_box_time/t_box, v_box, 0, 0, "N/A");
        }

        // 10. BOX FILTER - SIMD + Sliding Window
        {
            Image test_box(width, height);

            box_filter_simd_sliding_window(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_simd_sliding_window(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);
            print_row("Box - SIMD + Sliding Window", t_box, serial_box_time/t_box, v_box, 0, 0, "N/A");
        }

        cout << "---------------------------------------------------------------------------------------------------------------" << endl;
        cout << "TILED + SIMD IMPLEMENTATIONS" << endl;
        cout << "---------------------------------------------------------------------------------------------------------------" << endl;

        // 11. BOX FILTER - SIMD + Tiled
        {
            Image test_box(width, height);
            Image test_gauss(width, height);

            box_filter_simd_tiled(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_simd_tiled(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);
            
            gaussian_filter_simd_tiled(input, kernel_size, sigma);
            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_simd_tiled(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);
            
            print_row("Box - SIMD + Tiled", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        }

        // 12. BOX FILTER - SIMD Optimized + Tiled
        {
            Image test_box(width, height);
            Image test_gauss(width, height);

            box_filter_simd_optimized_tiled(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_simd_optimized_tiled(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);
            
            gaussian_filter_simd_optimized_tiled(input, kernel_size, sigma);
            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_simd_optimized_tiled(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);
            
            print_row("Box - SIMD Optimized + Tiled", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        }

        // 13. BOX FILTER - OpenMP SIMD Auto + Tiled
        {
            Image test_box(width, height);
            Image test_gauss(width, height);

            box_filter_openmp_simd_auto_tiled(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_openmp_simd_auto_tiled(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);
            
            gaussian_filter_openmp_simd_auto_tiled(input, kernel_size, sigma);
            double t_gauss = measure_time([&]() { 
                test_gauss = gaussian_filter_openmp_simd_auto_tiled(input, kernel_size, sigma); 
                prevent_dce(test_gauss);
            });
            string v_gauss = verify_result(ref_gauss, test_gauss);
            
            print_row("Box - OpenMP SIMD Auto + Tiled", t_box, serial_box_time/t_box, v_box, t_gauss, serial_gauss_time/t_gauss, v_gauss);
        }
        
        // 14. BOX FILTER - Sliding Window + Tiled
        {
            Image test_box(width, height);

            box_filter_sliding_window_tiled(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_sliding_window_tiled(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);
            print_row("Box - Sliding Window + Tiled", t_box, serial_box_time/t_box, v_box, 0, 0, "N/A");
        }

        // 15. BOX FILTER - SIMD + Sliding Window + Tiled
        {
            Image test_box(width, height);

            box_filter_simd_sliding_window_tiled(input, kernel_size);
            double t_box = measure_time([&]() { 
                test_box = box_filter_simd_sliding_window_tiled(input, kernel_size); 
                prevent_dce(test_box);
            });
            string v_box = verify_result(ref_box, test_box);
            print_row("Box - SIMD + Slide + Tiled", t_box, serial_box_time/t_box, v_box, 0, 0, "N/A");
        }

        cout << "---------------------------------------------------------------------------------------------------------------" << endl;
    }

    // ========================================================================
    // MODE 1, 2, 3: SINGLE STAGE (For Valgrind/Profiling)
    // ========================================================================
    else if (mode == 1) { // Serial
        double t = measure_time([&]() { 
            Image res = box_filter_serial(input, kernel_size); prevent_dce(res);
        });
        cout << "Serial Box Time: " << t << "s" << endl;
        
        measure_time([&]() { 
            Image res = gaussian_filter_serial(input, kernel_size, sigma); prevent_dce(res);
        });
    }
    else if (mode == 2) { // OpenMP (Using Static as representative, can be changed)
        double t = measure_time([&]() { 
            Image res = box_filter_openmp_static(input, kernel_size); prevent_dce(res);
        });
        cout << "OpenMP Box Time: " << t << "s" << endl;

        measure_time([&]() { 
            Image res = gaussian_filter_openmp_static(input, kernel_size, sigma); prevent_dce(res);
        });
    }
    else if (mode == 3) { // Tiled
        double t = measure_time([&]() { 
            Image res = box_filter_tiled(input, kernel_size); prevent_dce(res);
        });
        cout << "Tiled Box Time: " << t << "s" << endl;

        measure_time([&]() { 
            Image res = gaussian_filter_tiled(input, kernel_size, sigma); prevent_dce(res);
        });
    }
    else if (mode == 4) {
        // Run Baseline (Serial)
        Image res_s(width, height);
        double t_serial = measure_time([&]() { res_s = box_filter_serial(input, kernel_size); prevent_dce(res_s); });

        // Run OMP Static (Warmup + Measure)
        Image res_o(width, height);
        box_filter_openmp_static(input, kernel_size); 
        double t_omp = measure_time([&]() { res_o = box_filter_openmp_static(input, kernel_size); prevent_dce(res_o); });

        // Run Tiled (Warmup + Measure)
        Image res_t(width, height);
        box_filter_tiled(input, kernel_size);
        double t_tiled = measure_time([&]() { res_t = box_filter_tiled(input, kernel_size); prevent_dce(res_t); });

        // CSV Format: Threads, SerialTime, OMPTime, TiledTime
        cout << num_threads << "," << t_serial << "," << t_omp << "," << t_tiled << endl;
    }

    // ========================================================================
    // MODES 6-15: ISOLATED IMPLEMENTATIONS (For Cachegrind)
    // ========================================================================
    // Note: We create new Image objects inside scope to ensure fresh memory allocation
    // which simulates the "Cold Cache" scenario Valgrind is good at analyzing.
    
    // --- BASELINE VERSIONS ---
    else if (mode == 2) { // OMP Static
        Image res(width, height);
        measure_time([&]() { res = box_filter_openmp_static(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 6) { // SIMD Manual
        Image res(width, height);
        measure_time([&]() { res = box_filter_simd(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 7) { // SIMD Optimized
        Image res(width, height);
        measure_time([&]() { res = box_filter_simd_optimized(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 8) { // OMP SIMD Auto
        Image res(width, height);
        measure_time([&]() { res = box_filter_openmp_simd_auto(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 9) { // Sliding Window
        Image res(width, height);
        measure_time([&]() { res = box_filter_sliding_window(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 10) { // SIMD + Sliding
        Image res(width, height);
        measure_time([&]() { res = box_filter_simd_sliding_window(input, kernel_size); prevent_dce(res); });
    }

    // --- TILED VERSIONS ---
    else if (mode == 3) { // Tiled
        Image res(width, height);
        measure_time([&]() { res = box_filter_tiled(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 11) { // SIMD Manual + Tiled
        Image res(width, height);
        measure_time([&]() { res = box_filter_simd_tiled(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 12) { // SIMD Optimized + Tiled
        Image res(width, height);
        measure_time([&]() { res = box_filter_simd_optimized_tiled(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 13) { // OMP SIMD Auto + Tiled
        Image res(width, height);
        measure_time([&]() { res = box_filter_openmp_simd_auto_tiled(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 14) { // Sliding Window + Tiled
        Image res(width, height);
        measure_time([&]() { res = box_filter_sliding_window_tiled(input, kernel_size); prevent_dce(res); });
    }
    else if (mode == 15) { // SIMD + Sliding + Tiled
        Image res(width, height);
        measure_time([&]() { res = box_filter_simd_sliding_window_tiled(input, kernel_size); prevent_dce(res); });
    }

    return 0;
}