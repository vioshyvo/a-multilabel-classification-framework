#include "pca.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::RowMajor;
using Eigen::RowVectorXf;
using Eigen::SparseMatrix;
using Eigen::VectorXf;
using Eigen::VectorXi;


void Mrpt::queries(const Matrix<float, Dynamic, Dynamic, RowMajor> &Q, const int k,
                   int votes_required, int *out, float *out_distances) const {

  float *dist = out_distances;

  #pragma omp parallel for
  for (int i = 0; i < Q.rows(); ++i) {
    if (out_distances) dist = out_distances + i * k;
    query(Q.row(i), k, votes_required, out + i * k, dist);
  }
}


void Mrpt::exact_searches(const Matrix<float, Dynamic, Dynamic, RowMajor> &Q, const int k,
                          int *out, float *out_distances) const {

  float *dist = out_distances;

  #pragma omp parallel for
  for (int i = 0; i < Q.rows(); ++i) {
    if (out_distances) dist = out_distances + i * k;
    exact_knn(Q.row(i), k, std::vector<int>(), true, out + i * k, dist);
  }
}


int Mrpt::query(const VectorXf &q, const int k, int votes_required,
                int *out, float *out_distances, int *out_n_elected) const {

  VectorXi found_leaves(_n_trees);
  _query_paths(q, found_leaves.data(), false);
  std::vector<int> elected;

  int max_leaf_size = _n_samples / (1 << _depth) + 1;
  elected.reserve(_n_trees * max_leaf_size);

  const int step = 1 << (_max_depth - _depth);
  if (_voting_hashmap) {
    std::unordered_map<int, int> votes(_n_samples);
    for (int n_tree = 0; n_tree < _n_trees; ++n_tree) {
      int *start = _raw_leaves[n_tree] + _leaf_sizes(found_leaves(n_tree) * step, n_tree);
      const int *end = _raw_leaves[n_tree] + _leaf_sizes((found_leaves(n_tree) + 1) * step, n_tree);
      for (; start != end; ++start) {
        if (++votes[*start] == votes_required) {
          elected.push_back(*start);
        }
      }
    }
  } else {
    VectorXf votes = VectorXf::Zero(_n_samples);
    for (int n_tree = 0; n_tree < _n_trees; ++n_tree) {
      int *start = _raw_leaves[n_tree] + _leaf_sizes(found_leaves(n_tree) * step, n_tree);
      const int *end = _raw_leaves[n_tree] + _leaf_sizes((found_leaves(n_tree) + 1) * step, n_tree);
      for (; start != end; ++start) {
        if (++votes(*start) == votes_required) {
          elected.push_back(*start);
        }
      }
    }
  }

  if (out_n_elected) {
    *out_n_elected = elected.size();
  }

  return exact_knn(q, k, elected, false, out, out_distances);
}


int Mrpt::exact_knn(const VectorXf &q, const int k, const std::vector<int> &indices,
                    const bool all, int *out, float *out_distances) const {

  const int n_elected = all ? _n_samples : indices.size();
  if (n_elected == 0) return 0;
  const int n_return = std::min(k, n_elected);

  VectorXf distances(n_elected);

  if (all) {
    #pragma omp parallel for
    for (int i = 0; i < _n_samples; ++i)
      distances(i) = (_X.col(i) - q).squaredNorm();
  } else {
    #pragma omp parallel for
    for (int i = 0; i < n_elected; ++i)
      distances(i) = (_X.col(indices[i]) - q).squaredNorm();
  }

  if (k == 1) {
    MatrixXf::Index index;
    distances.minCoeff(&index);
    out[0] = all ? index : indices[index];
    return 1;
  }

  VectorXi idx(n_elected);
  std::iota(idx.data(), idx.data() + n_elected, 0);
  std::nth_element(idx.data(), idx.data() + n_return, idx.data() + n_elected,
                   [&distances](const int i1, const int i2)
                   { return distances(i1) < distances(i2); });

  if (all) {
    memcpy(out, idx.data(), n_return * sizeof(int));
  } else {
    for (int i = 0; i < n_return; ++i)
      out[i] = indices[idx(i)];
  }

  if (out_distances) {
    std::partial_sort(distances.data(), distances.data() + n_return, distances.data() + n_elected);

    for (int i = 0; i < n_return; ++i)
      out_distances[i] = std::sqrt(distances(i));
  }

  return n_return;
}


/**
 * Get paths through all the leaves for a single query.
 */
void Mrpt::_query_paths(const VectorXf &q, int *leaves, const bool all_paths) const {
  // The following loop goes over all trees, and routes the query to
  // exactly one leaf in each.
  const int tgt = static_cast<int>(_density * _dim);
  for (int n_tree = 0; n_tree < _n_trees; ++n_tree) {
    int idx_tree = 0;
    const int jj = n_tree * ((1 << _max_depth) - 1);
    for (int d = 0; d < _depth; ++d) {
      float proj = 0;
      const int *x = _random_dims[jj + idx_tree].data();
      const float *y = _random_vectors[jj + idx_tree].data();
      for (int i = tgt; i--; ++x, ++y) proj += q[*x] * *y;
      if (proj <= _split_points(idx_tree, n_tree)) {
        idx_tree = 2 * idx_tree + 1;
      } else {
        idx_tree = 2 * idx_tree + 2;
      }

      if (all_paths) {
        leaves[n_tree * _depth + d] = idx_tree - (1 << (d + 1)) + 1;
      }
    }

    if (!all_paths) {
      leaves[n_tree] = idx_tree - (1 << _depth) + 1;
    }
  }
}
