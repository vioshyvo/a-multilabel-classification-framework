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
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

struct Mrpt_node {
  float split_point;
  int split_dim;
  size_t left;
  size_t right;
};

int depth2n_0(int n, int depth) {
  int x = n / (1 << depth);
  return n % (1 << depth) ? x + 1 : x;
}

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
    * @param X_ Eigen ref to the data set, stored as one data point per column
    */
    Mrpt(const Eigen::Ref<const Eigen::MatrixXf> &X_) :
        X(Eigen::Map<const Eigen::MatrixXf>(X_.data(), X_.rows(), X_.cols())),
        n_corpus(X_.cols()),
        dim(X_.rows()) {}

    /**
    * @param X_ a float array containing the data set with each data point
    * stored contiguously in memory
    * @param dim_ dimension of the data
    * @param n_samples_ number of data points
    */
    Mrpt(const float *X_, int dim_, int n_samples_) :
        X(Eigen::Map<const Eigen::MatrixXf>(X_, dim_, n_samples_)),
        n_corpus(n_samples_),
        dim(dim_) {}

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
        throw std::out_of_range("The depth must belong to the set {1, ... , log2(n)}.");
      }

      if (density_ < -1.0001 || density_ > 1.0001 || (density_ > -0.9999 && density_ < -0.0001)) {
        throw std::out_of_range("The density must be on the interval (0,1].");
      }

      n_trees = n_trees_;
      depth = depth_;
      b = b_;
      n_pool = n_trees_ * depth_;

      if (density_ < 0) {
        density = 1.0 / std::sqrt(dim);
      } else {
        density = density_;
      }

      n_0 = depth2n_0(n_train, depth);

      const Eigen::Map<const Eigen::MatrixXi> knn(knn_.data(), knn_.rows(), knn_.cols());
      const Eigen::Map<const Eigen::MatrixXf> train(train_.data(), train_.rows(), train_.cols());

      trees = std::vector<std::vector<Mrpt_node>>(n_trees);
      labels_all = std::vector<std::vector<std::vector<int>>>(n_trees);
      votes_all = std::vector<std::vector<std::vector<int>>>(n_trees);

      #pragma omp parallel for
      for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
        std::vector<int> indices(n_train);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device gen;
        std::mt19937 r(gen());

        grow_subtree(0, n_train, indices, 0, n_tree, train, knn, r,
                     trees[n_tree], labels_all[n_tree], votes_all[n_tree]);
      }
    }

    /**@}*/


    /**@}*/

    /** @name Approximate k-nn search
    * A query using a non-autotuned index. Finds k approximate nearest neighbors
    * from a data set X for a query point q. Because the index is not autotuned,
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
          throw std::out_of_range("k must belong to the set {1, ..., n}.");
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
        std::vector<size_t> found_leaves(n_trees);
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
          const std::vector<Mrpt_node> &tree = trees[n_tree];
          Mrpt_node node = tree[0];
          while (node.right) {
            if (q(node.split_dim) <= node.split_point) {
              node = tree[node.left];
            } else {
              node = tree[node.right];
            }
          }
          found_leaves[n_tree] = node.left;
        }
        double end_traversal = omp_get_wtime();
        traversal_time = end_traversal - start_traversal;

        double start_voting = omp_get_wtime();
        std::vector<int> elected;
        Eigen::VectorXi votes = Eigen::VectorXi::Zero(n_corpus);
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
          const std::vector<int> &labels = labels_all[n_tree][found_leaves[n_tree]];
          const std::vector<int> &node_votes = votes_all[n_tree][found_leaves[n_tree]];

          const int n_labels = labels.size();
          for(int i = 0; i < n_labels; ++i) {
            const int label = labels[i];
            if((votes(label) += node_votes[i]) >= vote_threshold) {
              elected.push_back(label);
              votes(label) = std::numeric_limits<int>::min();
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
    * query point q from a data set X_. The indices of k nearest neighbors are
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

      const Eigen::Map<const Eigen::MatrixXf> X(X_data, dim, n_corpus);
      const Eigen::Map<const Eigen::VectorXf> q(q_data, dim);

      if (k < 1 || k > n_corpus) {
        throw std::out_of_range("k must be positive and no greater than the sample size of data X.");
      }

      Eigen::VectorXf distances(n_corpus);

      #pragma omp parallel for
      for (int i = 0; i < n_corpus; ++i)
        distances(i) = (X.col(i) - q).squaredNorm();

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
    * @param X Eigen ref to a data set
    * @param k number of neighbors searched for
    * @param out output buffer (size = k) for the indices of k nearest neighbors
    * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
    */
    static void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q,
                          const Eigen::Ref<const Eigen::MatrixXf> &X,
                          int k, int *out, float *out_distances = nullptr) {
      Mrpt::exact_knn(q.data(), X.data(), X.rows(), X.cols(), k, out, out_distances);
    }

    /**
    * @param q pointer to an array containing the query point
    * @param k number of neighbors searched for
    * @param out output buffer (size = k) for the indices of k nearest neighbors
    * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
    */
    void exact_knn(const float *q, int k, int *out, float *out_distances = nullptr) const {
      Mrpt::exact_knn(q, X.data(), dim, n_corpus, k, out, out_distances);
    }

    /**
    * @param q pointer to an array containing the query point
    * @param k number of points searched for
    * @param out output buffer (size = k) for the indices of k nearest neighbors
    * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
    */
    void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q, int k, int *out,
        float *out_distances = nullptr) const {
      Mrpt::exact_knn(q.data(), X.data(), dim, n_corpus, k, out, out_distances);
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

   std::pair<std::vector<int>,std::vector<int>> count_votes(int leaf_begin, int leaf_end,
                                                              const std::vector<int> &indices,
                                                              const Eigen::Map<const Eigen::MatrixXi> &knn) {
     int k_build = knn.rows();
     std::unordered_map<int,int> votes;
     for (int i = leaf_begin; i < leaf_end; ++i) {
       const Eigen::VectorXi knn_crnt = knn.col(indices[i]);
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
    size_t grow_subtree(int begin, int end, std::vector<int> &indices,
                        int tree_level, int n_tree,
                        const Eigen::Ref<const Eigen::MatrixXf> &train,
                        const Eigen::Map<const Eigen::MatrixXi> &knn,
                        std::mt19937 &rand_gen, std::vector<Mrpt_node> &tree,
                        std::vector<std::vector<int>> &labels,
                        std::vector<std::vector<int>> &votes) {
      const int n = end - begin;
      const int k_build = knn.rows();

      if (n <= n_0) {
        auto ret = count_votes(begin, end, indices, knn);
        size_t l = labels.size();
        labels.push_back(ret.first);
        votes.push_back(ret.second);
        size_t c_this = tree.size();
        tree.push_back(Mrpt_node {0, 0, l, 0});
        return c_this;
      }

      Eigen::VectorXi dims(dim);
      std::iota(dims.data(), dims.data() + dims.size(), 0);
      std::shuffle(dims.data(), dims.data() + dims.size(), rand_gen);
      int n_random_dim = density * dim;

      float max_gain = 0, max_split = 0;
      int max_dim = dims[0];

      Eigen::VectorXf left_entropies(n), right_entropies(n);

      const float *data = train.data();
      const int rows = train.rows();

      for (int l = 0; l < n_random_dim; ++l) {
        const int d = dims(l);
        std::sort(&indices[begin], &indices[end],
                  [data, rows, d](int i1, int i2) { return data[i1 * rows + d] < data[i2 * rows + d]; });

        // std::unordered_map<int, int> votes;
        std::vector<int> votes(n_corpus, 0);

        float entropy = 0;
        for (int i = 0; i < n; ++i) {
          const Eigen::VectorXi knn_crnt = knn.col(indices[begin + i]);
          for (int j = 0; j < k_build; ++j) {
            int v = ++votes[knn_crnt(j)];
            if (v > 1) entropy -= (v - 1) * log2(v - 1);
            entropy += v * log2(v);
          }
          left_entropies[i] = k_build * log2(i + 1) - entropy / (i + 1);
        }

        for (int i = 0; i < n - 1; ++i) {
          const Eigen::VectorXi knn_crnt = knn.col(indices[begin + i]);
          for (int j = 0; j < k_build; ++j) {
            int v = --votes[knn_crnt(j)];
            entropy -= (v + 1) * log2(v + 1);
            if (v) entropy += v * log2(v);
          }
          right_entropies[i] = k_build * log2(n - i - 1) - entropy / (n - i - 1);
        }
        right_entropies[n - 1] = 0;

        for (int i = 0; i < n; ++i) {
          if (i < n - 1 && train(d, indices[begin + i]) == train(d, indices[begin + i + 1]))
            continue;
          float left = static_cast<float>(i + 1) / n * left_entropies[i];
          float right = static_cast<float>(n - i - 1) / n * right_entropies[i];
          float gain = left_entropies[n - 1] - (left + right);
          if (gain >= max_gain) {
            max_gain = gain;
            max_dim = d;
            max_split = train(d, indices[begin + i]);
          }
        }
      }

      auto mid_point = std::partition(&indices[begin], &indices[end],
          [data, rows, max_dim, max_split](const int em) { return data[em * rows + max_dim] <= max_split; });
      int mid = static_cast<int>(mid_point - &indices[0]);

      if (mid == begin || mid == end) {
        auto ret = count_votes(begin, end, indices, knn);
        size_t l = labels.size();
        labels.push_back(ret.first);
        votes.push_back(ret.second);
        size_t c_this = tree.size();
        tree.push_back(Mrpt_node {0, 0, l, 0});
        return c_this;
      }

      size_t c_this = tree.size();
      tree.push_back(Mrpt_node {max_split, max_dim, 0, 0});
      size_t idx_left = grow_subtree(begin, mid, indices, tree_level + 1, n_tree,
                                       train, knn, rand_gen, tree, labels, votes);
      size_t idx_right = grow_subtree(mid, end, indices, tree_level + 1, n_tree,
                                        train, knn, rand_gen, tree, labels, votes);
      tree[c_this].left = idx_left;
      tree[c_this].right = idx_right;
      return c_this;
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
        distances(i) = (X.col(indices[i]) - q).squaredNorm();

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


    const Eigen::Map<const Eigen::MatrixXf> X; // training data
    std::vector<std::vector<Mrpt_node>> trees; // a vector of trees stored as vectors
    std::vector<std::vector<std::vector<int>>> labels_all;
    std::vector<std::vector<std::vector<int>>> votes_all;

    const int n_corpus; // sample size of data
    const int dim; // dimension of data
    int n_trees = 0; // number of RP-trees
    int depth = 0; // depth of an RP-tree with median split
    float density = -1.0; // expected ratio of non-zero components in a projection matrix
    int n_pool = 0; // amount of random vectors needed for all the RP-trees
    int b = 0;
    int n_0; // maximum leaf size of a tree
};

#endif // CPP_MRPT_H_
