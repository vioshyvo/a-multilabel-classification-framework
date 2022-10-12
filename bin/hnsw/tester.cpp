#include <vector>
#include <set>
#include <cstdio>
#include <stdint.h>
#include <omp.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>

#include "common.h"

int main(int argc, char **argv) {
    char *path = argv[1];
    size_t dim = atoi(argv[2]);
    int M = atoi(argv[3]);
    int efC = atoi(argv[4]);
    size_t n, n_test, n_validation;

    float *train_data = get_data((std::string(path) + "/corpus.bin").c_str(), dim, &n);
    float *test_data = get_data((std::string(path) + "/test.bin").c_str(), dim, &n_test);
    float *validation_data = get_data((std::string(path) + "/validation.bin").c_str(), dim, &n_validation);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexHNSWFlat index(dim, M);

    index.hnsw.efConstruction = efC;

    double ind_start = omp_get_wtime();
    index.train(n, train_data);
    index.add(n, train_data);
    double ind_end = omp_get_wtime();
    double it = ind_end - ind_start;

    omp_set_num_threads(1);
    for (int ind = 0; Ks[ind] != -1; ++ind) {
      int k = Ks[ind];

      for (int j = 5; j < argc; ++j) {
        std::vector<double> times, times2;
        std::vector<std::set<int>> idx, idx2;

        index.hnsw.efSearch = atoi(argv[j]);
        for (int i = 0; i < n_test; ++i) {
          float *dist = new float[k];
          std::vector<long> result(k);
          double start = omp_get_wtime();
          index.search(1, &test_data[i * dim], k, dist, &result[0]);
          double end = omp_get_wtime();
          times.push_back(end - start);
          idx.emplace_back(result.begin(), result.begin() + k);
        }

        for (int i = 0; i < n_validation; ++i) {
          float *dist = new float[k];
          std::vector<long> result(k);
          double start = omp_get_wtime();
          index.search(1, &validation_data[i * dim], k, dist, &result[0]);
          double end = omp_get_wtime();
          times2.push_back(end - start);
          idx2.emplace_back(result.begin(), result.begin() + k);
        }

        std::cout << k << " " << M << " " << efC << " " << argv[j] << " " << it << " ";
        results(k, times, idx, (std::string(path) + "/truth_" + std::to_string(k)).c_str());
        std::cout << " ";
        results(k, times2, idx2, (std::string(path) + "/validation_" + std::to_string(k)).c_str());
        std::cout << " ";
        recall_frequencies(k, idx, (std::string(path) + "/truth_" + std::to_string(k)).c_str());
        std::cout << std::endl;
      }
    }

    delete[] train_data;
    delete[] test_data;
    delete[] validation_data;

    return 0;
}
