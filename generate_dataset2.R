args <- commandArgs(TRUE)

if(length(args) != 4 & length(args) != 5) 
  stop('Usage: generate_dataset2.R <current-directory> <n_corpus> <n_train> <n_test> <seed?>')

# crnt_dir <- "/Users/DJKesoil/git/mrpt_dev/test_code"
crnt_dir <- args[1]
n_corpus <- as.integer(args[2])
n_train <- as.integer(args[3])
n_test <- as.integer(args[4])
if(length(args) == 5) 
  set.seed(as.integer(args[5]))

n_validation <- n_test
outdir <- file.path(crnt_dir, 'raw_data')
dir.create(outdir, showWarnings = FALSE)

xlim <- 10
dim <- 500

centroid_train <- 0
sd_trains <- c(0.5, 1, 2.5, 5)

source(file.path(crnt_dir, 'tools/data_sets.R'))

# dir.create(file.path(crnt_dir, 'fig'), showWarnings = FALSE)
# pdf(file.path(crnt_dir, 'fig', 'random2_data.pdf'), width = 7, height = 7)
# par(mfrow = c(2,2), mar = c(2,2,2,2))
for(sd_train in sd_trains) {
  outfname <- paste0('random2_sd', gsub('.', '_', sd_train, fixed = TRUE))
  dir1 <- file.path(outdir, outfname)
  dir.create(dir1, showWarnings = FALSE)

  cat('Generating', outfname, '...\n')
  corpus <- generate_unif(n_corpus, dim, xlim)
  train <- generate_norm(n_train, dim, centroid_train, sd_train, xlim)
  validation <- generate_norm(n_validation, dim, centroid_train, sd_train, xlim)
  test <- generate_norm(n_test, dim, centroid_train, sd_train, xlim)
  
  # plot(corpus[1, ], corpus[2, ], pch = 16, col = 'red', cex = .5,  asp = 1, bty = 'n',
  #      xlab = 'x', ylab = 'y', main = substitute(paste(sigma, ' = ', sd), list(sd = sd_train)))
  # points(train[1, ], train[2, ], pch = 16, col = 'purple', cex = .5)
  
  cat('Writing', outfname, 'to csv...\n')
  write_csv(t(corpus), file.path(dir1, 'corpus.csv'))
  write_csv(t(train), file.path(dir1, 'train.csv'))
  write_csv(t(validation), file.path(dir1, 'validation.csv'))
  write_csv(t(test), file.path(dir1, 'test.csv'))

  dimension_file <- file(file.path(dir1, 'dimensions.sh'))
  writeLines(c('#!/usr/bin/env bash',
               '',
               paste0('N_CORPUS=', n_corpus),
               paste0('N_TRAIN=', n_train),
               paste0('N_VALIDATION=', n_validation),
               paste0('N_TEST=', n_test),
               paste0('DIM=', dim)), dimension_file)
  close(dimension_file)
}
# dev.off()

