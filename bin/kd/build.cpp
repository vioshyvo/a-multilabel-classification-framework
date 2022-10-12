#include "kd.h"

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
  _split_dims = MatrixXi((1 << _depth) - 1, _n_trees);

  // allocate space for temporarily storing the leaves during building
  _raw_leaves = new int*[_n_trees];

  #pragma omp parallel for
  for (int n_tree = 0; n_tree < _n_trees; ++n_tree) {
    std::random_device rd;
    std::minstd_rand gen(rd());
    std::uniform_int_distribution<int> uni_dist(0, 4);
    _leaf_sizes(0, n_tree) = 0;
    _raw_leaves[n_tree] = new int[_n_samples];
    std::iota(_raw_leaves[n_tree], _raw_leaves[n_tree] + _n_samples, 0);
    _grow_subtree(_raw_leaves[n_tree], _n_samples, 0, 0, n_tree, gen, uni_dist);
  }
}


/**
 * Builds a single random projection tree. The tree is constructed by recursively
 * projecting the data onto a random vector and splitting into two by the median.
 */
void Mrpt::_grow_subtree(int *ind_ptr, const int n, const int tree_level, const int idx, const int n_tree,
                         std::minstd_rand &gen, std::uniform_int_distribution<int> &uni_dist) {

  if (tree_level == _depth) {
    const int leaf_idx = idx - (1 << _depth) + 2;
    _leaf_sizes(leaf_idx, n_tree) = _leaf_sizes(leaf_idx - 1, n_tree) + n;
    return;
  }

  const int idx_left = 2 * idx + 1;
  const int idx_right = idx_left + 1;
  const int split_point = (n % 2 == 1) ? n / 2 : n / 2 - 1;  // median split

  {
    VectorXf mu = VectorXf::Zero(_dim);
    VectorXf var = VectorXf::Zero(_dim);
    const int cnt = std::min(101, n);
    const float icnt = 1.0 / (float) cnt;
    for (int j = 0; j < cnt; ++j) {
      const float *dat = _X.col(ind_ptr[j]).data();
      for (int i = 0; i < _dim; ++i) {
        mu(i) += dat[i];
      }
    }
    for (int i = 0; i < _dim; ++i) mu(i) *= icnt;
    for (int j = 0; j < cnt; ++j) {
      const float *dat = _X.col(ind_ptr[j]).data();
      for (int i = 0; i < _dim; ++i) {
        const float y = dat[i] - mu(i);
        var(i) += y * y;
      }
    }

    const float *varptr = var.data();

    int num = 0;
    int topind[5];

    for (size_t i = 0; i < _dim; ++i) {
      if ((num < 5) || (varptr[i] > varptr[topind[num-1]])) {
        if (num < 5) {
          topind[num++] = i;
        }
        else {
          topind[num-1] = i;
        }
        int j = num - 1;
        while (j > 0 && varptr[topind[j]] > varptr[topind[j-1]]) {
          std::swap(topind[j], topind[j-1]);
          --j;
        }
      }
    }

    int ddd = topind[uni_dist(gen)];
    _split_dims(idx, n_tree) = ddd;

    // partition the current indices around the split point
    std::nth_element(ind_ptr, ind_ptr + split_point, ind_ptr + n,
              [this, &ddd](const int i1, const int i2) { return _X(ddd, i1) < _X(ddd, i2); });

    _split_points(idx, n_tree) = _X(ddd, ind_ptr[split_point]);
  }

  _grow_subtree(ind_ptr, split_point + 1, tree_level + 1, idx_left, n_tree, gen, uni_dist);
  _grow_subtree(ind_ptr + split_point + 1, n - split_point - 1, tree_level + 1, idx_right, n_tree, gen, uni_dist);
}
