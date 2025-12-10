#!/bin/bash

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LOG_FILE="benchmark_results.log"
PROFILER="./profiler"

# Function to calculate percentage change for cache misses
calc_improvement() {
    awk "BEGIN {printf \"%.2f%%\", (1 - $1/$2)*100}"
}

# ==============================================================================
# PREPARATION
# ==============================================================================
echo ">>> Rebuilding Project..."
make clean > /dev/null
make all > /dev/null

# Initialize Log File
echo "======================================================================" | tee $LOG_FILE
echo " AUTOMATED BENCHMARK REPORT" | tee -a $LOG_FILE
echo " Date: $(date)" | tee -a $LOG_FILE
echo " System: $(hostname)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ==============================================================================
# PART 1: SPEEDUP SCALING (1 to 16 Threads)
# Running './profiler 0 {threads}' and capturing output
# ==============================================================================
echo "----------------------------------------------------------------------" | tee -a $LOG_FILE
echo " PART 1: THREAD SCALING TEST (1 to 16 Threads)" | tee -a $LOG_FILE
echo "----------------------------------------------------------------------" | tee -a $LOG_FILE

for t in {1..16}; do
    echo "" | tee -a $LOG_FILE
    echo ">>> RUNNING WITH $t THREADS:" | tee -a $LOG_FILE
    echo "------------------------------------------------" | tee -a $LOG_FILE
    
    # Run profiler in Mode 0 (Full Table) with t threads
    # Capture output to variable to print to both console and log
    $PROFILER 0 $t | tee -a $LOG_FILE
done

echo "" | tee -a $LOG_FILE
echo "----------------------------------------------------------------------" | tee -a $LOG_FILE
echo " End of Scaling Test" | tee -a $LOG_FILE
echo "----------------------------------------------------------------------" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ==============================================================================
# PART 2: CACHE IMPROVEMENT ANALYSIS (Tiled vs Non-Tiled)
# Using Valgrind on specific modes (defined in profiler.cpp)
# ==============================================================================
echo "----------------------------------------------------------------------" | tee -a $LOG_FILE
echo " [PHASE 2] CACHE OPTIMIZATION ANALYSIS (Valgrind)" | tee -a $LOG_FILE
echo " Action: Recompiling WITHOUT -march=native to prevent 'Illegal Instruction'..." | tee -a $LOG_FILE
echo "----------------------------------------------------------------------" | tee -a $LOG_FILE

# FORCE RECOMPILE FOR VALGRIND SAFETY
make clean > /dev/null
# We override CXXFLAGS to REMOVE -march=native, but keep -mavx2
make all CXXFLAGS="-std=c++11 -O3 -Wall -fopenmp -mavx2 -mfma" > /dev/null

echo " Metric: L1 Data Read Misses (D1mr)" | tee -a $LOG_FILE
echo "----------------------------------------------------------------------" | tee -a $LOG_FILE

# Helper function to compare two modes
run_cache_test() {
    BASE_MODE=$1
    TILED_MODE=$2
    NAME=$3

    echo "Running comparison: $NAME..."
    
    # Run Valgrind (Quietly)
    valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes --log-file=base.log $PROFILER $BASE_MODE 1 > /dev/null 2>&1
    valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes --log-file=tiled.log $PROFILER $TILED_MODE 1 > /dev/null 2>&1

    # Extract D1 Misses
    D1_BASE=$(grep "D1  misses:" base.log | head -n 1 | awk '{print $4}' | tr -d ',')
    D1_TILED=$(grep "D1  misses:" tiled.log | head -n 1 | awk '{print $4}' | tr -d ',')
    
    # Calculate Improvement
    IMP=$(calc_improvement $D1_TILED $D1_BASE)

    # Print Row
    printf "%-30s | %-15s | %-15s | %-10s\n" "$NAME" "$D1_BASE" "$D1_TILED" "$IMP" | tee -a $LOG_FILE
}

# Print Table Header
printf "%-30s | %-15s | %-15s | %-10s\n" "Implementation" "Base Misses" "Tiled Misses" "Improvement" | tee -a $LOG_FILE
echo "-------------------------------|-----------------|-----------------|-----------" | tee -a $LOG_FILE

# 1. OpenMP Static vs Tiled
run_cache_test 2 3 "OpenMP Static vs Tiled"

# 2. SIMD Manual
run_cache_test 6 11 "SIMD Manual"

# 3. SIMD Optimized
run_cache_test 7 12 "SIMD Optimized"

# 4. OpenMP SIMD Auto
run_cache_test 8 13 "OpenMP SIMD Auto"

# 5. Sliding Window
run_cache_test 9 14 "Sliding Window"

# 6. SIMD + Sliding
run_cache_test 10 15 "SIMD + Sliding"

# Cleanup
rm base.log tiled.log

echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo " EXPERIMENT COMPLETE. Full results in $LOG_FILE" | tee -a $LOG_FILE
rm cachegrind.out.*