#include <vector>
#include <cstdio>
#include <stdint.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <typeinfo>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "rf-class.h"
#include "common.h"

using namespace Eigen;


int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " corpus_file test_file dim k" << std::endl;
    return 1;
  }

  char *corpus_file = argv[1];
  char *train_file = argv[2];
  size_t dim = atoi(argv[3]);
  int k = atoi(argv[4]);
  size_t n_corpus, n_train;

  float *corpus_data = get_data(corpus_file, dim, &n_corpus);
  float *train_data = get_data(train_file, dim, &n_train);

  for (int i = 0; i < n_train; ++i) {
      std::vector<int> result(k);
      double start = omp_get_wtime();
      Mrpt::exact_knn(train_data + i * dim, corpus_data, dim, n_corpus, k, &result[0]);
      double end = omp_get_wtime();
      printf("%g\n", end - start);
      for (int j = 0; j < k; ++j) printf("%d ", result[j]);
      printf("\n");
  }


  delete[] corpus_data;
  delete[] train_data;

  return 0;
}
