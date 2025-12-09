# Compiler Settings
CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -fopenmp -mavx2 -mfma -march=native

# Targets
ORIG_TARGET = image_filter
PROF_TARGET = profiler

# Valgrind Command (Quiet output to stdio, log to file)
VALGRIND_CMD = valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes

# Default: Build both targets
all: $(ORIG_TARGET) $(PROF_TARGET)

# Build Original (with SIMD support)
$(ORIG_TARGET): proj.cpp simd_filters.cpp
	$(CXX) $(CXXFLAGS) proj.cpp simd_filters.cpp -o $(ORIG_TARGET)

# Build Profiler
$(PROF_TARGET): profiler.cpp proj.cpp
	$(CXX) $(CXXFLAGS) profiler.cpp -o $(PROF_TARGET)

# ---------------------------------------------------------
# COMMANDS
# ---------------------------------------------------------

# 1. Benchmark (Time & Speedup Table)
benchmark: $(PROF_TARGET)
	@echo ""
	@echo "=== RUNNING SPEEDUP BENCHMARK ==="
	./$(PROF_TARGET) 0

# 2. Automated Cache Profiling & Analysis
profile_cache: $(PROF_TARGET)
	@echo "==============================================================="
	@echo "  RUNNING CACHE PROFILING (This may take a minute...)"
	@echo "==============================================================="
	
	@# 1. Run Stage 2 (Baseline)
	@echo "-> [1/2] Profiling Stage 2 (OpenMP Baseline)..."
	@$(VALGRIND_CMD) --log-file=stage2_report.txt ./$(PROF_TARGET) 2 > /dev/null 2>&1
	
	@# 2. Run Stage 3 (Tiled)
	@echo "-> [2/2] Profiling Stage 3 (Tiled Optimization)..."
	@$(VALGRIND_CMD) --log-file=stage3_report.txt ./$(PROF_TARGET) 3 > /dev/null 2>&1
	
	@# 3. Analyze Results (Shell Scripting inside Make)
	@echo "-> Calculating improvements..."
	@# Extract D1 Misses (Level 1)
	@S2_D1=$$(grep "D1  misses:" stage2_report.txt | head -n 1 | awk '{print $$4}' | tr -d ','); \
	 S3_D1=$$(grep "D1  misses:" stage3_report.txt | head -n 1 | awk '{print $$4}' | tr -d ','); \
	 # Extract LLd Misses (Last Level) \
	 S2_LL=$$(grep "LLd misses:" stage2_report.txt | head -n 1 | awk '{print $$4}' | tr -d ','); \
	 S3_LL=$$(grep "LLd misses:" stage3_report.txt | head -n 1 | awk '{print $$4}' | tr -d ','); \
	 # Calculate Improvement % using awk \
	 D1_IMP=$$(awk "BEGIN {printf \"%.2f%%\", (1 - $$S3_D1/$$S2_D1)*100}"); \
	 LL_IMP=$$(awk "BEGIN {printf \"%.2f%%\", (1 - $$S3_LL/$$S2_LL)*100}"); \
	 # Print Table \
	 echo ""; \
	 echo "==============================================================="; \
	 echo "  CACHE OPTIMIZATION REPORT"; \
	 echo "==============================================================="; \
	 printf "%-15s | %-18s | %-18s | %-15s\n" "Metric" "Stage 2 (Base)" "Stage 3 (Tiled)" "Improvement"; \
	 echo "----------------|--------------------|--------------------|----------------"; \
	 printf "%-15s | %-18s | %-18s | %-15s\n" "L1 (D1) Misses" "$$S2_D1" "$$S3_D1" "$$D1_IMP"; \
	 printf "%-15s | %-18s | %-18s | %-15s\n" "LL Data Misses" "$$S2_LL" "$$S3_LL" "$$LL_IMP"; \
	 echo "==============================================================="; \
	 echo "Detailed logs saved to: stage2_report.txt, stage3_report.txt"

clean:
	rm -f $(ORIG_TARGET) $(PROF_TARGET) *.txt cachegrind.out.* *.o

.PHONY: all benchmark profile_cache clean