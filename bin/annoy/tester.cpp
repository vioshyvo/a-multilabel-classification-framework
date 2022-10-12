#include <vector>
#include <set>
#include <cstdio>
#include <stdint.h>
#include <omp.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "annoylib.h"
#include "kissrandom.h"

#include "common.h"

int main(int argc, char **argv) {
    char *path = argv[1];
    size_t dim = atoi(argv[2]);
    int n_trees = atoi(argv[3]);
    int last_arg = 3;
    size_t n, n_test, n_validation;

    float *train_data = get_data((std::string(path) + "/corpus.bin").c_str(), dim, &n);
    float *test_data = get_data((std::string(path) + "/test.bin").c_str(), dim, &n_test);
    float *validation_data = get_data((std::string(path) + "/validation.bin").c_str(), dim, &n_validation);


    double ind_start = omp_get_wtime();
    AnnoyIndex<int, float, Euclidean, Kiss64Random> index(dim);
    for (int i = 0; i < n; ++i)
      index.add_item(i, train_data + i * dim);
    index.build(n_trees);
    double ind_end = omp_get_wtime();
    double it = ind_end - ind_start;

    omp_set_num_threads(1);
    for (int ind = 0; Ks[ind] != -1; ++ind) {
      int k = Ks[ind];

      for (int j = last_arg + 1; j < argc; ++j) {
        std::vector<double> times, times2;
        std::vector<std::set<int>> idx, idx2;

        int search_k = atoi(argv[j]);
        for (int i = 0; i < n_test; ++i) {
          std::vector<float> dist;
          std::vector<int> result;
          double start = omp_get_wtime();
          index.get_nns_by_vector(test_data + i * dim, k, search_k, &result, &dist);
          double end = omp_get_wtime();
          times.push_back(end - start);
          idx.emplace_back(result.begin(), result.begin() + k);
        }

        for (int i = 0; i < n_validation; ++i) {
          std::vector<float> dist;
          std::vector<int> result;
          double start = omp_get_wtime();
          index.get_nns_by_vector(validation_data + i * dim, k, search_k, &result, &dist);
          double end = omp_get_wtime();
          times2.push_back(end - start);
          idx2.emplace_back(result.begin(), result.begin() + k);
        }

        std::cout << k << " " << n_trees << " " << search_k << " " << it << " ";
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
