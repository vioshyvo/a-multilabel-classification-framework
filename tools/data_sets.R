generate_unif <- function(n, dim, xlim) matrix(runif(n * dim, min = -xlim, max = xlim), nrow = dim, ncol = n)

generate_norm <- function(n, dim, mean, sd, xlim) {
  n_sim <- n * dim
  x2 <- NULL
  repeat {
    x <- rnorm(2 * n_sim, mean, sd)
    x2 <- c(x2, x[abs(x) < xlim])
    if(length(x2) >= n_sim)
      return(matrix(x2[1:n_sim], nrow = dim, ncol = n))
  }
}

generate_clusters_norm <- function(n_clusters, dim, centroids, sds, xlim) {
  ret <- generate_norm(n_clusters[1], dim, centroids[1], sds[1], xlim)
  for(i in 2:length(n_clusters))
    ret <- cbind(ret, generate_norm(n_clusters[i], dim, centroids[i], sds[i], xlim))
  ret
}

generate_circle <- function(n, dim, r, noise = 1) {
  ret <- matrix(generate_norm(n, dim, 0, 1, Inf), nrow = dim, ncol = n)
  normalized <- apply(ret, 2, function(col) r * col / sqrt(sum(col^2)))
  normalized + rnorm(length(normalized), 0, noise)
}

write_csv <- function(x, fname) write.table(x, file = fname, row.names = FALSE, col.names = FALSE, sep = ',')

