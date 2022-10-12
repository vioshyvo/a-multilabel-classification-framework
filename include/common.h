#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <utility>

using namespace std;

int Ks[] = {10, -1};

float *get_data(const char *file, size_t dim, size_t *n) {
    struct stat sb;
    stat(file, &sb);
    size_t N = sb.st_size / (sizeof(float) * dim);
    *n = N;

    float *data = new float[N * dim];

    FILE *fd;
    fd = fopen(file, "rb");
    fread(data, sizeof(float), N * dim, fd);
    fclose(fd);

    return data;
}

float *get_data(const char *file, size_t n, size_t dim) {
    float *data = new float[n * dim];

    struct stat sb;
    stat(file, &sb);

    if(sb.st_size < n * dim * sizeof(float)) {
        std::cerr << "Size of the file is " << sb.st_size << ", while the requested size is: " << n * dim * sizeof(float) << "\n";
        return NULL;
    }

    FILE *fd;
    if ((fd = fopen(file, "rb")) == NULL) {
        std::cerr << "Could not open file " << file << " for reading.\n";
        return NULL;
    }

    size_t read = fread(data, sizeof(float), n * dim, fd);
    if (read != n * dim) {
        std::cerr << "Expected size of the read was " << n * dim << ", but " << read << " was read.\n";
        return NULL;
    }

    fclose(fd);
    return data;
}


int *get_data_int(const char *file, size_t dim, size_t *n) {
    struct stat sb;
    stat(file, &sb);
    size_t N = sb.st_size / (sizeof(int) * dim);
    *n = N;

    int *data = new int[N * dim];

    FILE *fd;
    fd = fopen(file, "rb");
    fread(data, sizeof(int), N * dim, fd);
    fclose(fd);

    return data;
}


int *get_data_int(const char *file, size_t n, size_t dim) {
    int *data = new int[n * dim];

    struct stat sb;
    stat(file, &sb);

    if(sb.st_size < n * dim * sizeof(int)) {
        std::cerr << "Size of the file is " << sb.st_size << ", while the requested size is: " << n * dim * sizeof(int) << "\n";
        return NULL;
    }

    FILE *fd;
    if ((fd = fopen(file, "rb")) == NULL) {
        std::cerr << "Could not open file " << file << " for reading.\n";
        return NULL;
    }

    size_t read = fread(data, sizeof(int), n * dim, fd);
    if (read != n * dim) {
        std::cerr << "Expected size of the read was " << n * dim << ", but " << read << " was read.\n";
        return NULL;
    }

    fclose(fd);
    return data;
}


int *get_data_knn(const char *file, size_t n_train, size_t k_build, size_t k_max) {
    int *data = new int[k_build * n_train];

    FILE *fd;
    fd = fopen(file, "rb");
    for (int i = 0; i < n_train; ++i) {
      fread(data + i * k_build, sizeof(int), k_build, fd);
      fseek(fd, (k_max - k_build) * sizeof(int), SEEK_CUR);
    }
    fclose(fd);

    return data;
}


void results(int k, const vector<double> &times, const vector<set<int>> &idx, const char *truth) {
    double time;
    vector<set<int>> correct;

    ifstream fs(truth);
    for (int j = 0; fs >> time; ++j) {
        set<int> res;
        for (int i = 0; i < k; ++i) {
            int r;
            fs >> r;
            res.insert(r);
        }
        correct.push_back(res);
    }

    vector<pair<double, double>> results;

    double accuracy, total_time = 0, total_accuracy = 0;
    for (unsigned i = 0; i < times.size(); ++i) {
        set<int> intersect;
        set_intersection(correct[i].begin(), correct[i].end(), idx[i].begin(), idx[i].end(),
                         inserter(intersect, intersect.begin()));
        accuracy = intersect.size() / static_cast<double>(k);
        total_time += times[i];
        total_accuracy += accuracy;
        results.push_back(make_pair(times[i], accuracy));
    }

    double mean_accuracy = total_accuracy / results.size(), variance = 0;
    for (auto res : results)
        variance += (res.second - mean_accuracy) * (res.second - mean_accuracy);
    variance /= (results.size() - 1);

    cout << setprecision(5);
    cout << mean_accuracy << " " << variance << " " << total_time;
}


void recall_frequencies(int k, const vector<set<int>> &idx, const char *truth) {
    double time;
    vector<set<int>> correct;

    ifstream fs(truth);
    for (int j = 0; fs >> time; ++j) {
        set<int> res;
        for (int i = 0; i < k; ++i) {
            int r;
            fs >> r;
            res.insert(r);
        }
        correct.push_back(res);
    }

    vector<int> frequencies(k + 1, 0);
    for (unsigned i = 0; i < idx.size(); ++i) {
        set<int> intersect;
        set_intersection(correct[i].begin(), correct[i].end(), idx[i].begin(), idx[i].end(),
                         inserter(intersect, intersect.begin()));
        ++frequencies[intersect.size()];
    }

    for (int i = 0; i < frequencies.size(); ++i)
      cout << frequencies[i] << " ";
}
