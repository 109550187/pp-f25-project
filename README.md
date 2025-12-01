# pp-f25-project
Group project of parallel image filtering implementation exploring multi-level optimization techniques including OpenMP scheduling strategies (static, dynamic, guided), SIMD vectorization, and cache-aware blocking for box and Gaussian filters.

# Building
```bash
# Clone the repository
git clone https://github.com/109550187/pp-f25-project.git
cd pp-f25-project

# Build
make

# Clean build files
make clean
```

## Usage
### Basic Usage
```bash
# Run with default settings (1 thread, 4096×4096 image)
./image_filter

# Run with 4 threads
./image_filter 4

# Run with 8 threads
./image_filter 8
```

### Advanced Configuration

Edit `proj.cpp` to change:
```cpp
// Line ~320
int width = 4096;       // Image width (try: 1024, 2048, 4096, 8192)
int height = 4096;      // Image height
int kernel_size = 7;    // Filter kernel size (try: 3, 7, 15)
float sigma = 2.0f;     // Gaussian sigma (spread)
```

Then rebuild:
```bash
make clean && make
```
