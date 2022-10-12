CC=gcc
CXX=$(shell if command -v g++-8 >/dev/null; then echo g++-8; else echo g++; fi)
EIGEN_PATH=../../lib
INCLUDE_PATH=../../include
CFLAGS+=-std=c++14 -Ofast -march=native -DNDEBUG -flto -fopenmp -I./ -I$(EIGEN_PATH) -I$(INCLUDE_PATH)
LDFLAGS+=-lgomp

all:
	$(CXX) -o tester build.cpp search.cpp tester.cpp $(CFLAGS) $(LDFLAGS)
