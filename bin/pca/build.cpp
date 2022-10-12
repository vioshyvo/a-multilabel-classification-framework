#include "pca.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <unordered_map>
#include <iostream>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::PermutationMatrix;
using Eigen::RowMajor;
using Eigen::RowVectorXf;
using Eigen::SparseMatrix;
using Eigen::VectorXf;
using Eigen::VectorXi;


void Mrpt::grow(const int n_trees, const int depth, const float density) {
  _grow(n_trees, depth, density);
  _max_depth = depth;
}


/**
 * The actual build process.
 */
void Mrpt::_grow(const int n_trees, const int depth, const float density) {
  _n_trees = n_trees;
  _depth = depth;
  _density = density;
  _n_pool = ((1 << _depth) - 1) * _n_trees;

  _split_points = MatrixXf((1 << _depth) - 1, _n_trees);
  _leaf_sizes = MatrixXi((1 << _depth) + 1, _n_trees);

  _random_dims = std::vector<VectorXi>();
  _random_vectors = std::vector<VectorXf>();
  _random_dims.reserve(_n_pool);
  _random_vectors.reserve(_n_pool);

  int tgt = static_cast<int>(density * _dim);
  for (int i = 0; i < _n_pool; ++i) {
    _random_dims.emplace_back(tgt);
    _random_vectors.emplace_back(tgt);
  }

  // allocate space for temporarily storing the leaves during building
  _raw_leaves = new int*[_n_trees];

  #pragma omp parallel for
  for (int n_tree = 0; n_tree < _n_trees; ++n_tree) {
    std::random_device rd;
    std::minstd_rand gen(rd());
    std::uniform_int_distribution<int> uni_dist(0, _dim - 1);
    std::normal_distribution<float> norm_dist(0, 1);

    for (int i = ((1 << _depth) - 1) * n_tree; i < ((1 << _depth) - 1) * (n_tree + 1); ++i) {
      std::generate(_random_dims[i].data(),
                    _random_dims[i].data() + tgt,
                    [&uni_dist, &gen] { return uni_dist(gen); });
      std::generate(_random_vectors[i].data(),
                    _random_vectors[i].data() + tgt,
                    [&norm_dist, &gen] { return norm_dist(gen); });
    }

    _leaf_sizes(0, n_tree) = 0;

    _raw_leaves[n_tree] = new int[_n_samples];
    std::iota(_raw_leaves[n_tree], _raw_leaves[n_tree] + _n_samples, 0);
    _grow_subtree(_raw_leaves[n_tree], _n_samples, 0, 0, n_tree);
  }
}


/**
 * Builds a single random projection tree. The tree is constructed by recursively
 * projecting the data onto a random vector and splitting into two by the median.
 */
void Mrpt::_grow_subtree(int *ind_ptr, const int n, const int tree_level, const int idx, const int n_tree) {

  if (tree_level == _depth) {
    const int leaf_idx = idx - (1 << _depth) + 2;
    _leaf_sizes(leaf_idx, n_tree) = _leaf_sizes(leaf_idx - 1, n_tree) + n;
    return;
  }

  const int idx_left = 2 * idx + 1;
  const int idx_right = idx_left + 1;
  const int split_point = (n & 1) ? n / 2 : n / 2 - 1;  // median split

  {
    VectorXi dims = _random_dims[n_tree * ((1 << _depth) - 1) + idx];
    VectorXf rv = _random_vectors[n_tree * ((1 << _depth) - 1) + idx];
    rv /= rv.norm();

    MatrixXf tmp = _X(dims, Eigen::Map<VectorXi>(ind_ptr, n));
    float isz = 1. / (n - 1);
    MatrixXf centered = tmp.colwise() - tmp.rowwise().mean();
    MatrixXf cov = 2 * 0.01 * isz * (centered * centered.transpose());

    rv /= rv.norm();
    for (int i = 0; i < 20; ++i) {
      VectorXf last = rv;
      rv += cov * rv;
      rv /= rv.norm();
      if ((rv - last).cwiseAbs().mean() < 0.01) break;
    }

    VectorXf data = rv.transpose() * tmp;
    _random_vectors[n_tree * ((1 << _depth) - 1) + idx] = rv;

    std::unordered_map<int, int> inv_idx;
    for (int i = 0; i < n; ++i) {
      inv_idx[ind_ptr[i]] = i;
    }

    // partition the current indices around the split point
    std::nth_element(ind_ptr, ind_ptr + split_point, ind_ptr + n,
                    [&data, &inv_idx](const int i1, const int i2) { return data[inv_idx[i1]] < data[inv_idx[i2]]; });

    _split_points(idx, n_tree) = data[inv_idx[ind_ptr[split_point]]];
  }

  _grow_subtree(ind_ptr, split_point + 1, tree_level + 1, idx_left, n_tree);
  _grow_subtree(ind_ptr + split_point + 1, n - split_point - 1, tree_level + 1, idx_right, n_tree);
}
