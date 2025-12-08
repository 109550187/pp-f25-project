#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip> // For std::setw (table formatting)
#include <string>  // For to_string

// 1. Define the flag to DISABLE the main() inside proj.cpp
#define PROFILER_MODE

// 2. Import all functions and structs from proj.cpp
#include "proj.cpp"

using namespace std;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// This function tricks the compiler. 
// 1. It reads data (preventing optimization/dead code elimination).
// 2. It casts 'd' to void (preventing the "unused variable" warning).
void prevent_dce(const Image& res) {
    volatile float d = res.data[0];
    (void)d; // <--- This silences the warning!
}

// Helper to print a table row
void print_row(string name, double t_box, double s_box, double t_gauss, double s_gauss) {
    cout << left << setw(24) << name 
         << fixed << setprecision(4) 
         << setw(10) << t_box << "s " 
         << setw(10) << "(" + to_string(s_box).substr(0,4) + "x)   |   "
         << setw(10) << t_gauss << "s "
         << setw(10) << "(" + to_string(s_gauss).substr(0,4) + "x)" 
         << endl;
}

// ============================================================================
// MAIN PROFILER
// ============================================================================

int main(int argc, char* argv[]) {
    // Mode: 0=All(Benchmark), 1=Serial, 2=OpenMP(Stage2), 3=Tiled(Stage3)
    int mode = 0;
    if (argc > 1) mode = atoi(argv[1]);

    // Setup Data
    int width = 4096;
    int height = 4096;
    int kernel_size = 7;
    float sigma = 2.0f;

    Image input = create_test_image(width, height);

    // Variables to store baseline times for speedup calculation
    static double serial_box_time = 0.0;
    static double serial_gauss_time = 0.0;

    // ========================================================================
    // MODE 0: RUN EVERYTHING & COMPARE (For "make benchmark")
    // ========================================================================
    if (mode == 0) {
        cout << "Running Performance Benchmark (4096 x 4096 image)..." << endl;
        cout << "-----------------------------------------------------------------------------------" << endl;
        cout << left << setw(24) << "Version" 
             << setw(26) << "Box Filter (Time/Speedup)" 
             << "   |   " 
             << setw(26) << "Gaussian Filter (Time/Speedup)" << endl;
        cout << "-----------------------------------------------------------------------------------" << endl;

        // 1. SERIAL
        serial_box_time = measure_time([&]() { 
            Image res = box_filter_serial(input, kernel_size); 
            prevent_dce(res); // Used helper here
        });
        serial_gauss_time = measure_time([&]() { 
            Image res = gaussian_filter_serial(input, kernel_size, sigma); 
            prevent_dce(res);
        });
        print_row("Serial (Baseline)", serial_box_time, 1.0, serial_gauss_time, 1.0);

        // 2. OPENMP - STATIC
        double static_box = measure_time([&]() { 
            Image res = box_filter_openmp_static(input, kernel_size); 
            prevent_dce(res);
        });
        double static_gauss = measure_time([&]() { 
            Image res = gaussian_filter_openmp_static(input, kernel_size, sigma); 
            prevent_dce(res);
        });
        print_row("OpenMP (Static)", static_box, serial_box_time/static_box, static_gauss, serial_gauss_time/static_gauss);

        // 3. OPENMP - DYNAMIC
        double dyn_box = measure_time([&]() { 
            Image res = box_filter_openmp_dynamic(input, kernel_size); 
            prevent_dce(res);
        });
        double dyn_gauss = measure_time([&]() { 
            Image res = gaussian_filter_openmp_dynamic(input, kernel_size, sigma); 
            prevent_dce(res);
        });
        print_row("OpenMP (Dynamic)", dyn_box, serial_box_time/dyn_box, dyn_gauss, serial_gauss_time/dyn_gauss);

        // 4. OPENMP - GUIDED
        double guid_box = measure_time([&]() { 
            Image res = box_filter_openmp_guided(input, kernel_size); 
            prevent_dce(res);
        });
        double guid_gauss = measure_time([&]() { 
            Image res = gaussian_filter_openmp_guided(input, kernel_size, sigma); 
            prevent_dce(res);
        });
        print_row("OpenMP (Guided)", guid_box, serial_box_time/guid_box, guid_gauss, serial_gauss_time/guid_gauss);

        // 5. TILED (Stage 3)
        double tiled_box = measure_time([&]() { 
            Image res = box_filter_tiled(input, kernel_size); 
            prevent_dce(res);
        });
        double tiled_gauss = measure_time([&]() { 
            Image res = gaussian_filter_tiled(input, kernel_size, sigma); 
            prevent_dce(res);
        });
        print_row("Tiled (Stage 3)", tiled_box, serial_box_time/tiled_box, tiled_gauss, serial_gauss_time/tiled_gauss);
        
        cout << "-----------------------------------------------------------------------------------" << endl;
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
    else if (mode == 2) { // OpenMP (Using Static as representative)
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