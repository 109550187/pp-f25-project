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

## Testing

### Quick Test
Run a quick test with different configurations:
```bash
./quick_test.sh
```

### Comprehensive Testing
Run full test suite (tests different thread counts, image sizes, and kernel sizes):
```bash
./test_all.sh
```

Test results will be saved in `test_results/` directory.

### Manual Testing
```bash
# Test with specific parameters: threads width height kernel_size
./image_filter 4 2048 2048 7

# All implementations are automatically verified for correctness
# Look for [OK] or [FAIL] markers in the output
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Features

- ✅ Serial baseline implementations
- ✅ OpenMP parallelization (static, dynamic, guided scheduling)
- ✅ SIMD vectorization (manual AVX2 and compiler auto-vectorization)
- ✅ Automatic correctness verification
- ✅ Performance benchmarking and reporting
