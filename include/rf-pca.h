#ifndef CPP_MRPT_H_
#define CPP_MRPT_H_

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/SparseCore>


class Mrpt {
 public:
    /** @name Constructors
    * The constructor does not actually build the index. The building is done
    * by the function grow() which has to be called before queries can be made.
    * There are two different versions of the constructor which differ only
    * by the type of the input data. The first version takes the data set
    * as `Ref` to `MatrixXf`, which means that the argument
    * can be either `MatrixXf` or `Map<MatrixXf>` (also certain blocks of `MatrixXf`
    * may be accepted, see [Eigen::Ref](https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html)
    * for more information). The second version takes a float
    * pointer to an array containing the data set, and the dimension and
    * the sample size of the data. There are also corresponding versions
    * of all the member functions which take input data. In all cases the data
    * is assumed to be stored in column-major order such that each data point
    * is stored contiguously in memory. In all cases no copies are made of
    * the original data matrix. */

    /**
    * @param corpus_ Eigen ref to the corpus, stored as one data point per column
    */
    Mrpt(const Eigen::Ref<const Eigen::MatrixXf> &corpus_) :
        corpus(Eigen::Map<const Eigen::MatrixXf>(corpus_.data(), corpus_.rows(), corpus_.cols())),
        n_corpus(corpus_.cols()),
        dim(corpus_.rows()) {}

    /**@}*/

    /** @name Normal index building.
    * Build a normal (not autotuned) index.
    */

    /**
    * Build a normal index.
    *
    * @param n_trees_ number of trees to be grown
    * @param depth_ depth of the trees; in the set
    * \f$\{1,2, \dots ,\lfloor \log_2 (n) \rfloor \}\f$, where \f$n \f$ is the number
    * of data points
    * @param knn_ Eigen ref to the knn matrix of training set; a column is a training set point,
    * and a row is k:th neighbor
    * @param train_ Eigen ref to the training set; a column is a training set point
    * @param density_ expected proportion of non-zero components in the
    * random vectors; on the interval \f$(0,1]\f$; default value sets density to
    * \f$ 1 / \sqrt{d} \f$, where \f$d\f$ is the dimension of the data
    * @param seed seed given to a rng when generating random vectors;
    * a default value 0 initializes the rng randomly with std::random_device
    */
    void grow(int n_trees_, int depth_, const Eigen::Ref<const Eigen::MatrixXi> &knn_,
              const Eigen::Ref<const Eigen::MatrixXf> &train_, float density_ = -1.0, int b_ = 0, int seed = 0) {

      if (!empty()) {
        throw std::logic_error("The index has already been grown.");
      }

      if (n_trees_ <= 0) {
        throw std::out_of_range("The number of trees must be positive.");
      }

      int n_train = train_.cols();
      if (depth_ <= 0 || depth_ > std::log2(n_train)) {
        throw std::out_of_range("The depth must belong to the set {1, ... , log2(n_train)}.");
      }

      if (density_ < -1.0001 || density_ > 1.0001 || (density_ > -0.9999 && density_ < -0.0001)) {
        throw std::out_of_range("The density must be on the interval (0,1].");
      }

      n_trees = n_trees_;
      depth = depth_;
      n_inner_nodes = (1 << depth_) - 1;
      n_leaves = 1 << depth_;
      b = b_;
      n_pool = n_inner_nodes * n_trees;
      n_array = 1 << (depth_ + 1);


      if (density_ < 0) {
        density = 1.0 / std::sqrt(dim);
      } else {
        density = density_;
      }

      const Eigen::Map<const Eigen::MatrixXi> knn(knn_.data(), knn_.rows(), knn_.cols());
      const Eigen::Map<const Eigen::MatrixXf> train(train_.data(), train_.rows(), train_.cols());

      split_points = Eigen::MatrixXf(n_array, n_trees);
      labels_all = std::vector<std::vector<std::vector<int>>>(n_trees);
      votes_all = std::vector<std::vector<std::vector<int>>>(n_trees);

      _random_dims = std::vector<Eigen::VectorXi>();
      _random_vectors = std::vector<Eigen::VectorXf>();
      _random_dims.reserve(n_pool);
      _random_vectors.reserve(n_pool);

      int tgt = static_cast<int>(density * dim);
      for (int i = 0; i < n_pool; ++i) {
        _random_dims.emplace_back(tgt);
        _random_vectors.emplace_back(tgt);
      }

      #pragma omp parallel for
      for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
        labels_all[n_tree] = std::vector<std::vector<int>>(n_leaves);
        votes_all[n_tree] = std::vector<std::vector<int>>(n_leaves);
        std::vector<int> indices(n_train);
        std::iota(indices.begin(), indices.end(), 0);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> uni_dist(0, dim - 1);
        std::normal_distribution<float> norm_dist(0, 1);

        for (int i = n_inner_nodes * n_tree; i < n_inner_nodes * (n_tree + 1); ++i) {
          std::generate(_random_dims[i].data(),
                        _random_dims[i].data() + tgt,
                        [&uni_dist, &gen] { return uni_dist(gen); });
          std::generate(_random_vectors[i].data(),
                        _random_vectors[i].data() + tgt,
                        [&norm_dist, &gen] { return norm_dist(gen); });
        }

        grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, labels_all[n_tree], votes_all[n_tree], knn, train);
      }
    }

    /**@}*/


    /**@}*/

    /** @name Approximate k-nn search
    * A query using a non-autotuned index. Finds k approximate nearest neighbors
    * from a data set corpus for a query point q. Because the index is not autotuned,
    * k and vote threshold are set manually. The indices of k nearest neighbors
    * are written to a buffer out, which has to be preallocated to have at least
    * length k. Optionally also Euclidean distances to these k nearest points
    * are written to a buffer out_distances. If there are less than k points in
    * the candidate set, -1 is written to the remaining locations of the
    * output buffers.
    */

    /**
    * Approximate k-nn search using a normal index.
    *
    * @param data pointer to an array containing the query point
    * @param k number of nearest neighbors searched for
    * @param vote_threshold - number of votes required for a query point to be included in the candidate set
    * @param out output buffer (size = k) for the indices of k approximate nearest neighbors
    * @param out_distances optional output buffer (size = k) for distances to k approximate nearest neighbors
    * @param out_n_elected optional output parameter (size = 1) for the candidate set size
    */
    void query(const float *data, int k, int vote_threshold, int *out,
               double &vote_time, double &projection_time, double &traversal_time, double &exact_time,
               float *out_distances = nullptr, int *out_n_elected = nullptr) const {

        if (k <= 0 || k > n_corpus) {
          throw std::out_of_range("k must belong to the set {1, ..., n_corpus}.");
        }

        if (vote_threshold <= 0) {
          throw std::out_of_range("vote_threshold must be positive");
        }

        if (empty()) {
          throw std::logic_error("The index must be built before making queries.");
        }

        const Eigen::Map<const Eigen::VectorXf> q(data, dim);
        projection_time = 0;

        double start_traversal = omp_get_wtime();
        std::vector<int> found_leaves(n_trees);
        const int tgt = static_cast<int>(density * dim);
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
          int idx_tree = 0;
          const int jj = n_tree * ((1 << depth) - 1);
          for (int d = 0; d < depth; ++d) {
            float proj = 0;
            const int *x = _random_dims[jj + idx_tree].data();
            const float *y = _random_vectors[jj + idx_tree].data();
            for (int i = tgt; i--; ++x, ++y) proj += q[*x] * *y;
            if (proj <= split_points(idx_tree, n_tree)) {
              idx_tree = 2 * idx_tree + 1;
            } else {
              idx_tree = 2 * idx_tree + 2;
            }
          }
          found_leaves[n_tree] = idx_tree - n_inner_nodes;
        }
        double end_traversal = omp_get_wtime();
        traversal_time = end_traversal - start_traversal;

        double start_voting = omp_get_wtime();
        std::vector<int> elected;
        Eigen::VectorXi votes_total = Eigen::VectorXi::Zero(n_corpus);

        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
          int leaf_idx = found_leaves[n_tree];
          const std::vector<int> &labels = labels_all[n_tree][leaf_idx];
          const std::vector<int> &votes = votes_all[n_tree][leaf_idx];
          int n_labels = labels.size();
          for(int i = 0; i < n_labels; ++i) {
            if((votes_total(labels[i]) += votes[i]) >= vote_threshold) {
              elected.push_back(labels[i]);
              votes_total(labels[i]) = std::numeric_limits<int>::min();
            }
          }
        }
        double end_voting = omp_get_wtime();
        vote_time = end_voting - start_voting;

        if (out_n_elected)
          *out_n_elected = elected.size();

        double start_exact = omp_get_wtime();
        exact_knn(q, k, elected, out, out_distances);
        double end_exact = omp_get_wtime();
        exact_time = end_exact - start_exact;
    }

    /**
    *  Approximate k-nn search using a normal index.
    *
    * @param q Eigen ref to the query point
    * @param k number of nearest neighbors searched for
    * @param vote_threshold number of votes required for a query point to be included in the candidate set
    * @param out output buffer (size = k) for the indices of k approximate nearest neighbors
    * @param out_distances optional output buffer (size = k) for distances to k approximate nearest neighbors
    * @param out_n_elected optional output parameter (size = 1) for the candidate set size
    */
    void query(const Eigen::Ref<const Eigen::VectorXf> &q, int k, int vote_threshold, int *out,
               double &vote_time, double &projection_time, double &traversal_time, double &exact_time,
               float *out_distances = nullptr, int *out_n_elected = nullptr) const {
      query(q.data(), k, vote_threshold, out, vote_time, projection_time, traversal_time, exact_time, out_distances, out_n_elected);
    }

    /**@}*/


    /** @name Exact k-nn search
    * Functions for fast exact k-nn search: find k nearest neighbors for a
    * query point q from a data set corpus_. The indices of k nearest neighbors are
    * written to a buffer out, which has to be preallocated to have at least
    * length k. Optionally also the Euclidean distances to these k nearest points
    * are written to a buffer out_distances. There are both static and member
    * versions.
    */

    /**
    * @param q_data pointer to an array containing the query point
    * @param X_data pointer to an array containing the data set
    * @param dim dimension of data
    * @param n_corpus number of points in a data set
    * @param k number of neighbors searched for
    * @param out output buffer (size = k) for the indices of k nearest neighbors
    * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
    */
    static void exact_knn(const float *q_data, const float *X_data, int dim, int n_corpus,
        int k, int *out, float *out_distances = nullptr) {

      const Eigen::Map<const Eigen::MatrixXf> corpus(X_data, dim, n_corpus);
      const Eigen::Map<const Eigen::VectorXf> q(q_data, dim);

      if (k < 1 || k > n_corpus) {
        throw std::out_of_range("k must be positive and no greater than the sample size of data corpus.");
      }

      Eigen::VectorXf distances(n_corpus);

      #pragma omp parallel for
      for (int i = 0; i < n_corpus; ++i)
        distances(i) = (corpus.col(i) - q).squaredNorm();

      if (k == 1) {
        Eigen::MatrixXf::Index index;
        distances.minCoeff(&index);
        out[0] = index;

        if (out_distances)
          out_distances[0] = std::sqrt(distances(index));

        return;
      }

      Eigen::VectorXi idx(n_corpus);
      std::iota(idx.data(), idx.data() + n_corpus, 0);
      std::partial_sort(idx.data(), idx.data() + k, idx.data() + n_corpus,
                       [&distances](int i1, int i2) { return distances(i1) < distances(i2); });

      for (int i = 0; i < k; ++i)
        out[i] = idx(i);

      if (out_distances) {
        for (int i = 0; i < k; ++i)
          out_distances[i] = std::sqrt(distances(idx(i)));
      }
    }

    /**
    * @param q Eigen ref to a query point
    * @param corpus Eigen ref to a data set
    * @param k number of neighbors searched for
    * @param out output buffer (size = k) for the indices of k nearest neighbors
    * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
    */
    static void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q,
                          const Eigen::Ref<const Eigen::MatrixXf> &corpus,
                          int k, int *out, float *out_distances = nullptr) {
      Mrpt::exact_knn(q.data(), corpus.data(), corpus.rows(), corpus.cols(), k, out, out_distances);
    }

    /**
    * @param q pointer to an array containing the query point
    * @param k number of neighbors searched for
    * @param out output buffer (size = k) for the indices of k nearest neighbors
    * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
    */
    void exact_knn(const float *q, int k, int *out, float *out_distances = nullptr) const {
      Mrpt::exact_knn(q, corpus.data(), dim, n_corpus, k, out, out_distances);
    }

    /**
    * @param q pointer to an array containing the query point
    * @param k number of points searched for
    * @param out output buffer (size = k) for the indices of k nearest neighbors
    * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
    */
    void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q, int k, int *out,
        float *out_distances = nullptr) const {
      Mrpt::exact_knn(q.data(), corpus.data(), dim, n_corpus, k, out, out_distances);
    }

    static Eigen::MatrixXi exact_all_pairs(const float *corpus, size_t n_corpus, size_t dim, size_t k, const float *training_data, size_t n_train) {
      Eigen::MatrixXi true_knn(k, n_train);
      for(size_t i = 0; i < n_train; ++i)
        Mrpt::exact_knn(training_data + i * dim, corpus, dim, n_corpus, k, true_knn.data() + i * k);
      return true_knn;
    }
    /**@}*/

    /** @name Utility functions
    */

    /**
    * Is the index is already constructed or not?
    *
    * @return - is the index empty?
    */
    bool empty() const {
      return n_trees == 0;
    }

    /**@}*/

 private:

   std::pair<std::vector<int>,std::vector<int>> count_votes(std::vector<int>::iterator leaf_begin,
                                                            std::vector<int>::iterator leaf_end,
                                                            const Eigen::Map<const Eigen::MatrixXi> &knn) {
     int k_build = knn.rows();
     std::unordered_map<int,int> votes;
     for (auto it = leaf_begin; it != leaf_end; ++it) {
       const Eigen::VectorXi knn_crnt = knn.col(*it);
       for (int j = 0; j < k_build; ++j)
         ++votes[knn_crnt(j)];
     }

     std::vector<int> out_labels;
     std::vector<int> out_votes;

     for (const auto &v : votes)
       if (v.second >= b) {
         out_labels.push_back(v.first);
         out_votes.push_back(v.second);
       }

     return {out_labels, out_votes};
   }

    /**
    * Builds a single random projection tree. The tree is constructed by recursively
    * projecting the data on a random vector and splitting into two by the median.
    */
    void grow_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                      int tree_level, int i, int n_tree,
                      std::vector<std::vector<int>> &labels_tree, std::vector<std::vector<int>> &votes_tree,
                      const Eigen::Map<const Eigen::MatrixXi> &knn,
                      const Eigen::Map<const Eigen::MatrixXf> &train) {
      int n = end - begin;
      int idx_left = 2 * i + 1;
      int idx_right = idx_left + 1;

      if (tree_level == depth) {
        int index_leaf = i - n_inner_nodes;
        auto ret = count_votes(begin, end, knn);
        labels_tree[index_leaf] = ret.first;
        votes_tree[index_leaf] = ret.second;
        return;
      }

      Eigen::VectorXi dims = _random_dims[n_tree * ((1 << depth) - 1) + i];
      Eigen::VectorXf rv = _random_vectors[n_tree * ((1 << depth) - 1) + i];
      rv /= rv.norm();

      Eigen::MatrixXf tmp = train(dims, Eigen::Map<Eigen::VectorXi>(&*begin, n));
      float isz = 1. / (n - 1);
      Eigen::MatrixXf centered = tmp.colwise() - tmp.rowwise().mean();
      Eigen::MatrixXf cov = 2 * 0.01 * isz * (centered * centered.transpose());

      rv /= rv.norm();
      for (int i = 0; i < 20; ++i) {
        Eigen::VectorXf last = rv;
        rv += cov * rv;
        rv /= rv.norm();
        if ((rv - last).cwiseAbs().mean() < 0.01) break;
      }

      Eigen::VectorXf data = rv.transpose() * tmp;
      _random_vectors[n_tree * ((1 << depth) - 1) + i] = rv;

      std::unordered_map<int, int> inv_idx;
      for (int i = 0; i < n; ++i) {
        inv_idx[*(begin + i)] = i;
      }

      std::nth_element(begin, begin + n / 2, end,
            [&data, &inv_idx](const int i1, const int i2) { return data[inv_idx[i1]] < data[inv_idx[i2]]; });

      auto mid = end - n / 2;

      if (n % 2) {
        split_points(i, n_tree) = data[inv_idx[*(mid - 1)]];
      } else {
        auto left_it = std::max_element(begin, mid,
            [&data, &inv_idx](const int i1, const int i2) { return data[inv_idx[i1]] < data[inv_idx[i2]]; });
        split_points(i, n_tree) = (data[inv_idx[*mid]] + data[inv_idx[*left_it]]) / 2.0;
      }

      grow_subtree(begin, mid, tree_level + 1, idx_left, n_tree, labels_tree, votes_tree, knn, train);
      grow_subtree(mid, end, tree_level + 1, idx_right, n_tree, labels_tree, votes_tree, knn, train);
    }

    /**
    * Find k nearest neighbors from data for the query point
    */
    void exact_knn(const Eigen::Map<const Eigen::VectorXf> &q, int k, const std::vector<int> &indices,
                   int *out, float *out_distances = nullptr) const {

      if (indices.empty()) {
        for (int i = 0; i < k; ++i)
          out[i] = -1;
        if (out_distances) {
          for (int i = 0; i < k; ++i)
            out_distances[i] = -1;
        }
        return;
      }

      int n_elected = indices.size();
      Eigen::VectorXf distances(n_elected);

      #pragma omp parallel for
      for (int i = 0; i < n_elected; ++i)
        distances(i) = (corpus.col(indices[i]) - q).squaredNorm();

      if (k == 1) {
        Eigen::MatrixXf::Index index;
        distances.minCoeff(&index);
        out[0] = n_elected ? indices[index] : -1;

        if (out_distances)
          out_distances[0] = n_elected ? std::sqrt(distances(index)) : -1;

        return;
      }

      int n_to_sort = n_elected > k ? k : n_elected;
      Eigen::VectorXi idx(n_elected);
      std::iota(idx.data(), idx.data() + n_elected, 0);
      std::partial_sort(idx.data(), idx.data() + n_to_sort, idx.data() + n_elected,
                       [&distances](int i1, int i2) { return distances(i1) < distances(i2); });

      for (int i = 0; i < k; ++i)
        out[i] = i < n_elected ? indices[idx(i)] : -1;

      if (out_distances) {
        for (int i = 0; i < k; ++i)
          out_distances[i] = i < n_elected ? std::sqrt(distances(idx(i))) : -1;
      }
    }


    const Eigen::Map<const Eigen::MatrixXf> corpus; // corpus from which nearest neighbors are searched
    Eigen::MatrixXf split_points; // all split points in all trees
    std::vector<std::vector<std::vector<int>>> labels_all;
    std::vector<std::vector<std::vector<int>>> votes_all;

    std::vector<Eigen::VectorXi> _random_dims;
    std::vector<Eigen::VectorXf> _random_vectors;

    const int n_corpus; // size of corpus
    const int dim; // dimension of data
    int n_trees = 0; // number of RP-trees
    int depth = 0; // depth of an RP-tree with median split
    float density = -1.0; // expected ratio of non-zero components in a projection matrix
    int n_pool = 0; // amount of random vectors needed for all the RP-trees
    int n_array = 0; // length of the one RP-tree as array
    int b = 0;
    int n_inner_nodes = 0;
    int n_leaves = 0;
};

#endif // CPP_MRPT_H_
