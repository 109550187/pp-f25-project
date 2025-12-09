#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string> 
#include <cmath> 

#define PROFILER_MODE

#include "proj.cpp"

using namespace std;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void prevent_dce(const Image& res) {
    volatile float d = res.data[0];
    (void)d; // <--- This silences the warning!
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

// Updated Helper to print a table row with Verification Status
void print_row(string name, double t_box, double s_box, string v_box, double t_gauss, double s_gauss, string v_gauss) {
    cout << left << setw(20) << name 
         << fixed << setprecision(4) 
         << setw(9) << t_box << "s " 
         << setw(8) << "(" + to_string(s_box).substr(0,4) + "x) "
         << setw(6) << v_box << " |   "
         << setw(9) << t_gauss << "s "
         << setw(8) << "(" + to_string(s_gauss).substr(0,4) + "x) " 
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

    // ------------------------------------------------------------------------
    // NEW: Handle Thread Count
    // ------------------------------------------------------------------------
    int num_threads = omp_get_max_threads(); // Default to max
    if (argc > 2) {
        num_threads = atoi(argv[2]);
        if (num_threads < 1) num_threads = 1;
        omp_set_num_threads(num_threads);
    }
    // ------------------------------------------------------------------------

    // Setup Data
    int width = 4096;
    int height = 4096;
    int kernel_size = 7;
    float sigma = 2.0f;

    Image input = create_test_image(width, height);

    // Store Reference Images
    Image ref_box(width, height);
    Image ref_gauss(width, height);

    static double serial_box_time = 0.0;
    static double serial_gauss_time = 0.0;

    if (mode == 0) {
        cout << "Running Performance Benchmark (4096 x 4096)..." << endl;
        cout << "Active Threads: " << omp_get_max_threads() << endl; 
        cout << "-------------------------------------------------------------------------------------------------" << endl;
        cout << left << setw(20) << "Version" 
             << setw(25) << "Box Filter (Time/Spd/Chk)" 
             << " |   " 
             << setw(25) << "Gaussian Filter (Time/Spd/Chk)" << endl;
        cout << "-------------------------------------------------------------------------------------------------" << endl;

        // 1. SERIAL (BASELINE & REFERENCE GENERATION)
        serial_box_time = measure_time([&]() { 
            ref_box = box_filter_serial(input, kernel_size); 
            prevent_dce(ref_box);
        });
        serial_gauss_time = measure_time([&]() { 
            ref_gauss = gaussian_filter_serial(input, kernel_size, sigma); 
            prevent_dce(ref_gauss);
        });
        print_row("Serial (Base)", serial_box_time, 1.0, "REF", serial_gauss_time, 1.0, "REF");

        // Temporary images to hold parallel results for verification
        Image test_box(width, height);
        Image test_gauss(width, height);

        // 2. OPENMP - STATIC
        double static_box = measure_time([&]() { 
            test_box = box_filter_openmp_static(input, kernel_size); 
            prevent_dce(test_box);
        });
        string v_box = verify_result(ref_box, test_box);

        double static_gauss = measure_time([&]() { 
            test_gauss = gaussian_filter_openmp_static(input, kernel_size, sigma); 
            prevent_dce(test_gauss);
        });
        string v_gauss = verify_result(ref_gauss, test_gauss);

        print_row("OpenMP (Static)", static_box, serial_box_time/static_box, v_box, static_gauss, serial_gauss_time/static_gauss, v_gauss);

        // 3. OPENMP - DYNAMIC
        double dyn_box = measure_time([&]() { 
            test_box = box_filter_openmp_dynamic(input, kernel_size); 
            prevent_dce(test_box);
        });
        v_box = verify_result(ref_box, test_box);

        double dyn_gauss = measure_time([&]() { 
            test_gauss = gaussian_filter_openmp_dynamic(input, kernel_size, sigma); 
            prevent_dce(test_gauss);
        });
        v_gauss = verify_result(ref_gauss, test_gauss);

        print_row("OpenMP (Dynamic)", dyn_box, serial_box_time/dyn_box, v_box, dyn_gauss, serial_gauss_time/dyn_gauss, v_gauss);

        // 4. OPENMP - GUIDED
        double guid_box = measure_time([&]() { 
            test_box = box_filter_openmp_guided(input, kernel_size); 
            prevent_dce(test_box);
        });
        v_box = verify_result(ref_box, test_box);

        double guid_gauss = measure_time([&]() { 
            test_gauss = gaussian_filter_openmp_guided(input, kernel_size, sigma); 
            prevent_dce(test_gauss);
        });
        v_gauss = verify_result(ref_gauss, test_gauss);

        print_row("OpenMP (Guided)", guid_box, serial_box_time/guid_box, v_box, guid_gauss, serial_gauss_time/guid_gauss, v_gauss);

        // 5. TILED (Stage 3)
        double tiled_box = measure_time([&]() { 
            test_box = box_filter_tiled(input, kernel_size); 
            prevent_dce(test_box);
        });
        v_box = verify_result(ref_box, test_box);

        double tiled_gauss = measure_time([&]() { 
            test_gauss = gaussian_filter_tiled(input, kernel_size, sigma); 
            prevent_dce(test_gauss);
        });
        v_gauss = verify_result(ref_gauss, test_gauss);

        print_row("Tiled (Stage 3)", tiled_box, serial_box_time/tiled_box, v_box, tiled_gauss, serial_gauss_time/tiled_gauss, v_gauss);
        
        cout << "-------------------------------------------------------------------------------------------------" << endl;
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

    return 0;
}