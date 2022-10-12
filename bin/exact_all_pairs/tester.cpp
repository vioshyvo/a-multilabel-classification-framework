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

template <typename T>
void write_memory(const T *mem, std::string spath, size_t nrow, size_t ncol) {
  const char *path = spath.c_str();
  FILE *fd;
  if ((fd = fopen(path, "wb")) == NULL) {
    std::cerr << "In file " << __FILE__ << ", line " << __LINE__ << ": " << path << " could not be opened for writing" << std::endl;
    return;
  }
  fwrite(mem, sizeof(T), nrow * ncol, fd);
  fclose(fd);
}


int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0] << " corpus_file train_file dim kmax data_path outfname" << std::endl;
    return 1;
  }

  char *corpus_file = argv[1];
  char *train_file = argv[2];
  size_t dim = atoi(argv[3]);
  int kmax = atoi(argv[4]);
  std::string data_path(argv[5]);
  if (!data_path.empty() && data_path.back() != '/')
    data_path += '/';
  std::string outfname(argv[6]);
  size_t n_corpus, n_train;

  float *corpus_data = get_data(corpus_file, dim, &n_corpus);
  float *train_data = get_data(train_file, dim, &n_train);

  double start = omp_get_wtime();
  Eigen::MatrixXi true_knn = Mrpt::exact_all_pairs(corpus_data, n_corpus, dim, kmax, train_data, n_train);
  double end = omp_get_wtime();
  std::cout << "All " << kmax  << "-nn for " << n_train << " points took "
            << (end - start) / 60 << " min." << std::endl << std::endl;

  write_memory(true_knn.data(), outfname, kmax, n_train);

  delete[] corpus_data;
  delete[] train_data;

  return 0;
}
