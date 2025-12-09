CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -fopenmp -mavx2 -mfma -march=native

TARGET = image_filter

# Build with proj.cpp and simd_filters.cpp
all: $(TARGET)

$(TARGET): proj.cpp simd_filters.cpp
	$(CXX) $(CXXFLAGS) proj.cpp simd_filters.cpp -o $(TARGET)
	@echo ""
	@echo "Build complete!"
	@echo ""
	@echo "Run with:"
	@echo "  ./$(TARGET)                          # Use default threads"
	@echo "  ./$(TARGET) 4 # Use 4 threads"
	@echo "  ./$(TARGET) 8 # Use 8 threads"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all run clean
