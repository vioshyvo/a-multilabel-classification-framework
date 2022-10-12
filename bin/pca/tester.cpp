#include <vector>
#include <cstdio>
#include <stdint.h>
#include <omp.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "pca.h"
#include <Eigen/Core>

#include "common.h"

using namespace Eigen;

int main(int argc, char **argv) {
    char *path = argv[1];
    size_t dim = atoi(argv[2]);
    int n_trees = atoi(argv[3]);
    int depth = atoi(argv[4]);
    float sparsity = atof(argv[5]);
    size_t n, n_test, n_validation;

    float *train_data = get_data((std::string(path) + "/corpus.bin").c_str(), dim, &n);
    float *test_data = get_data((std::string(path) + "/test.bin").c_str(), dim, &n_test);
    float *validation_data = get_data((std::string(path) + "/validation.bin").c_str(), dim, &n_validation);

    const Map<const MatrixXf> X = Map<const MatrixXf>(train_data, dim, n);
    Mrpt index(X);

    double ind_start = omp_get_wtime();
    index.grow(n_trees, depth, sparsity);
    double ind_end = omp_get_wtime();
    double it = ind_end - ind_start;

    omp_set_num_threads(1);
    for (int ind = 0; Ks[ind] != -1; ++ind) {
        int k = Ks[ind];

        for (int arg = 6; arg < argc; ++arg) {
            int votes = atoi(argv[arg]);
            if (votes > n_trees) continue;

            std::vector<double> times, times2;
            std::vector<std::set<int>> idx, idx2;
            int total_n_elected = 0;

            for (int i = 0; i < n_test; ++i) {
                std::vector<int> result(k);
                std::vector<float> distances(k);
                int n_elected = 0;
                Map<VectorXf> q = Map<VectorXf>(&test_data[i * dim], dim);
                double start = omp_get_wtime();
                int nn = index.query(q, k, votes, &result[0], &distances[0], &n_elected);
                double end = omp_get_wtime();
                times.push_back(end - start);
                idx.emplace_back(result.begin(), result.begin() + nn);
                total_n_elected += n_elected;
            }

            for (int i = 0; i < n_validation; ++i) {
                std::vector<int> result(k);
                Map<VectorXf> q = Map<VectorXf>(&validation_data[i * dim], dim);
                double start = omp_get_wtime();
                int nn = index.query(q, k, votes, &result[0]);
                double end = omp_get_wtime();
                times2.push_back(end - start);
                idx2.emplace_back(result.begin(), result.begin() + nn);
            }

            std::cout << k << " " << n_trees << " " << depth << " " << sparsity << " " << votes << " " << it << " ";
            results(k, times, idx, (std::string(path) + "/truth_" + std::to_string(k)).c_str());
            std::cout << " " << static_cast<double>(total_n_elected) / n_test << " ";
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
