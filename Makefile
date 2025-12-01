CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -fopenmp

TARGET = image_filter

all: $(TARGET)

$(TARGET): proj.cpp
	$(CXX) $(CXXFLAGS) proj.cpp -o $(TARGET)
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
	rm -f $(TARGET)

.PHONY: all run clean
