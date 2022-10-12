CC=$(shell if command -v g++-8 >/dev/null; then echo g++-8; else echo g++; fi)
CXX=g++
FAISS_PATH=../../lib/faiss
INCLUDE_PATH=../../include
CFLAGS+=-std=c++14 -Ofast -march=native -DNDEBUG -flto -fopenmp -I./ -I$(INCLUDE_PATH) -I$(FAISS_PATH)/lib/include -L$(FAISS_PATH)/lib/lib
LDFLAGS+=-lgomp -l:libfaiss.a -static-libstdc++ -lopenblas

all:
	$(CC) -o tester tester.cpp $(CFLAGS) $(LDFLAGS)
