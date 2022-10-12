#ifndef LIB_CPP_MRPT_H_
#define LIB_CPP_MRPT_H_

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vector>

using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::RowMajor;
using Eigen::RowVectorXf;
using Eigen::SparseMatrix;
using Eigen::VectorXf;
using Eigen::VectorXi;


class Mrpt {
 public:

  /**
   * The constructor of the index. The actual data structure is built by a
   * separate function 'grow' that has to be called before queries can be made.
   * @param data - An Eigen matrix containing the data.
   */
  explicit Mrpt(const Map<const MatrixXf> data) :
    _X(data),
    _n_samples(data.cols()),
    _dim(data.rows()),
    _n_trees(1),
    _depth(1),
    _density(1.),
    _n_pool(1),
    _voting_hashmap(false)
  {
  }


  ~Mrpt() {}


  inline int get_n_samples() const       { return _n_samples; }
  inline int get_dim() const             { return _dim; }
  inline int get_n_trees() const         { return _n_trees; }
  inline int get_depth() const           { return _depth; }
  inline float get_density() const       { return _density; }


  /**
  * The function that starts the actual index construction.
  * @param n_trees - The number of trees to be used in the index.
  * @param depth - The depth of the trees.
  * @param density - Expected ratio of non-zero components in a projection matrix.
  * @return
  */
  void grow(const int n_trees, const int depth, const float density);


  /**
   * Do multiple queries at once.
   * @param Q - A matrix of queries, one per each row.
   * @param k - The number of nearest neighbours to return.
   * @param votes_required - The number of votes required for an object to be included in the linear search step.
   * @param out - Output buffer for the indices of the k approximate nearest neighbours.
   * @param out_distances - Output buffer for distances of the k approximate nearest neighbours (optional).
   * @return
   */
  void queries(const Matrix<float, Dynamic, Dynamic, RowMajor> &Q, const int k, int votes_required,
               int *out, float *out_distances = nullptr) const;


  /**
   * Do multiple exact searches at once.
   * @param Q - A matrix of queries, one per each row.
   * @param k - The number of nearest neighbours to return.
   * @param out - Output buffer for the indices of the k approximate nearest neighbours.
   * @param out_distances - Output buffer for distances of the k approximate nearest neighbours (optional).
   * @return
   */
  void exact_searches(const Matrix<float, Dynamic, Dynamic, RowMajor> &Q, const int k,
                      int *out, float *out_distances = nullptr) const;


  /**
   * This function finds the k approximate nearest neighbours of the query object
   * q. The accuracy of the query depends on both the parameters used for index
   * construction and additional parameters given to this function.
   * @param q - The query object whose neighbours the function finds.
   * @param k - The number of nearest neighbours to return.
   * @param votes_required - The number of votes required for an object to be included in the linear search step.
   * @param out - Output buffer for the indices of the k approximate nearest neighbours.
   * @param out_distances - Output buffer for distances of the k approximate nearest neighbours (optional).
   * @return
   */
  int query(const VectorXf &q, const int k, int votes_required, int *out, float *out_distances = nullptr, int *out_n_elected = nullptr) const;


  /**
   * Find the k nearest neighbours from data for the query point using exact search.
   * @param q - The query object whose neighbours the function finds.
   * @param k - The number of nearest neighbours to return.
   * @param indices - Indices of the points in the original matrix where the search is made.
   * @param all - Use all samples as indices.
   * @param out - Output buffer for the indices of the k approximate nearest neighbours.
   * @param out_distances - Output buffer for distances of the k approximate nearest neighbours (optional).
   * @return
   */
  int exact_knn(const VectorXf &q, const int k, const std::vector<int> &indices, const bool all,
                int *out, float *out_distances = nullptr) const;


 private:

  void _grow(const int n_trees, const int depth, const float density);


  void _grow_subtree(int *ind_ptr, const int n, const int tree_level, const int i, const int n_tree);


  void _query_paths(const VectorXf &q, int *leaves, const bool all_paths) const;


  const Map<const MatrixXf>   _X;
  MatrixXf                    _split_points;
  std::vector<VectorXi>       _random_dims;
  std::vector<VectorXf>       _random_vectors;
  MatrixXi                    _leaf_sizes;
  int **                      _raw_leaves;

  int     _n_samples;
  int     _dim;
  int     _n_trees;
  int     _depth;
  float   _density;
  int     _n_pool;
  int     _max_depth;
  bool    _voting_hashmap;
};

#endif  // LIB_CPP_MRPT_H_
