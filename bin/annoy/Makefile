CC=$(shell if command -v g++-8 >/dev/null; then echo g++-8; else echo g++; fi)
INCLUDE_PATH=../../include
CFLAGS+=-std=c++14 -Ofast -march=native -DNDEBUG -flto -fopenmp -I./ -I$(INCLUDE_PATH)
LDFLAGS+=-lgomp -static-libstdc++

all:
	$(CC) -o tester tester.cpp $(CFLAGS) $(LDFLAGS)
