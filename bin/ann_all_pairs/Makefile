CXX=$(shell if command -v g++-8 >/dev/null; then echo g++-8; else echo g++; fi)
CXXFLAGS+=-std=c++14 -Ofast -march=native -DNDEBUG -flto -fopenmp
EIGEN_PATH=../../lib
INCLUDE_PATH=../../include


all: tester

tester.o: tester.cpp $(INCLUDE_PATH)/common.h $(INCLUDE_PATH)/rp.h
	$(CXX) -I$(EIGEN_PATH) -I$(INCLUDE_PATH) $(CXXFLAGS) -c tester.cpp

tester: tester.o
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: clean
clean:
	$(RM) tester *.o
