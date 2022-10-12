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
#include <faiss/IndexIVFPQ.h>

#include "common.h"

int main(int argc, char **argv) {
    char *path = argv[1];
    size_t dim = atoi(argv[2]);
    size_t n, n_test, n_validation;

    float *train_data = get_data((std::string(path) + "/corpus.bin").c_str(), dim, &n);
    float *test_data = get_data((std::string(path) + "/test.bin").c_str(), dim, &n_test);
    float *validation_data = get_data((std::string(path) + "/validation.bin").c_str(), dim, &n_validation);

    faiss::IndexFlatL2 coarse_quantizer(dim);
    int ncentroids = static_cast<int>(4*sqrt(n));
    faiss::IndexIVFPQ ivf(&coarse_quantizer, dim, ncentroids, 4, 8);
    faiss::IndexRefineFlat index(&ivf);

    double ind_start = omp_get_wtime();
    index.train(n, train_data);
    index.add(n, train_data);
    double ind_end = omp_get_wtime();
    double it = ind_end - ind_start;

    int probes[20], k_factors[20];

    int j = 3;
    for (; j < argc; ++j) {
      probes[j - 3] = atoi(argv[j]);
      if (probes[j - 3] == -1) break;
    }

    int l = ++j;
    for (; j < argc; ++j) {
      k_factors[j - l] = atoi(argv[j]);
      if (k_factors[j - l] == -1) break;
    }

    omp_set_num_threads(1);
    for (int ind = 0; Ks[ind] != -1; ++ind) {
      int k = Ks[ind];

      for (int j = 0; probes[j] != -1; ++j) {
        for (int l = 0; k_factors[l] != -1; ++l) {
          std::vector<double> times, times2;
          std::vector<std::set<int>> idx, idx2;

          ivf.nprobe = probes[j];
          index.k_factor = k_factors[l];

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

          std::cout << k << " "  << probes[j] << " " << k_factors[l] << " " << it << " ";
          results(k, times, idx, (std::string(path) + "/truth_" + std::to_string(k)).c_str());
          std::cout << " ";
          results(k, times2, idx2, (std::string(path) + "/validation_" + std::to_string(k)).c_str());
          std::cout << " ";
          recall_frequencies(k, idx, (std::string(path) + "/truth_" + std::to_string(k)).c_str());
          std::cout << std::endl;
        }
      }
    }

    delete[] train_data;
    delete[] test_data;
    delete[] validation_data;

    return 0;
}
