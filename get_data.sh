#!/usr/bin/env bash

. config.sh
REMOVE_DOWNLOADED=false # remove downloaded datasets after they've been converted
N_TEST=1000 # number of test queries
DATA_DIR="$BASE_DATA_DIR"
RAW_DATA_DIR=raw_data
SEED=12345


MNIST_DIR="$DATA_DIR/mnist"
RAW_MNIST_DIR="$RAW_DATA_DIR/mnist"

if [ ! -f "$MNIST_DIR/data.bin" ]; then
  mkdir -p "$MNIST_DIR"
  mkdir -p "$RAW_MNIST_DIR"
  echo "Downloading MNIST..."
  wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" -O "$RAW_MNIST_DIR/train-images-idx3-ubyte.gz"
  wget "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" -O "$RAW_MNIST_DIR/train-labels-idx1-ubyte.gz"
  echo "Extracting MNIST..."
  gunzip "$RAW_MNIST_DIR/train-images-idx3-ubyte.gz"
  gunzip "$RAW_MNIST_DIR/train-labels-idx1-ubyte.gz"
  echo "Converting MNIST..."
  python2 tools/binary_converter.py "$RAW_MNIST_DIR/train-images-idx3-ubyte" "$MNIST_DIR/data.bin"
fi

if [ ! -f "$MNIST_DIR/corpus.bin" ]; then
  python2 tools/binary_converter.py --sample "$MNIST_DIR/data.bin" "$MNIST_DIR/tmp.bin" "$MNIST_DIR/test.bin" $N_TEST 784
  python2 tools/binary_converter.py --sample "$MNIST_DIR/tmp.bin" "$MNIST_DIR/corpus.bin" "$MNIST_DIR/validation.bin" $N_TEST 784
  rm "$MNIST_DIR/tmp.bin"
  if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm "$RAW_MNIST_DIR/train-images-idx3-ubyte"
  fi
fi


RANDOM_N_CORPUS=100000
if [ ! -f "$DATA_DIR/random2_sd1/corpus.bin" ]; then
  if [ ! -f "$RAW_DATA_DIR/random2_sd1/corpus.csv" ]; then
    if ! command -v Rscript >/dev/null; then
      module load R/3.2.3-foss-2016b
    fi
    Rscript generate_dataset2.R $(pwd) $RANDOM_N_CORPUS $RANDOM_N_CORPUS $N_TEST $SEED
  fi

  for sd in 1 2_5 5; do
    RANDOM_NAME="random2_sd$sd"
    RANDOM_DIR="$DATA_DIR/$RANDOM_NAME"
    RAW_DIR="$RAW_DATA_DIR/$RANDOM_NAME"

    mkdir -p "$RANDOM_DIR"
    python2 tools/binary_converter.py "$RAW_DIR/corpus.csv" "$RANDOM_DIR/corpus.bin"
    python2 tools/binary_converter.py "$RAW_DIR/train.csv" "$RANDOM_DIR/train.bin"
    python2 tools/binary_converter.py "$RAW_DIR/validation.csv" "$RANDOM_DIR/validation.bin"
    python2 tools/binary_converter.py "$RAW_DIR/test.csv" "$RANDOM_DIR/test.bin"
  done
fi


FASHION_DIR="$DATA_DIR/fashion"
RAW_FASHION_DIR="$RAW_DATA_DIR/fashion"

if [ ! -f "$FASHION_DIR/data.bin" ]; then
  mkdir -p "$FASHION_DIR"
  mkdir -p "$RAW_FASHION_DIR"
  echo "Downloading fashion..."
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz" -O "$RAW_FASHION_DIR/train-images-idx3-ubyte.gz"
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz" -O "$RAW_FASHION_DIR/train-labels-idx1-ubyte.gz"
  echo "Extracting fashion..."
  gunzip "$RAW_FASHION_DIR/train-images-idx3-ubyte.gz"
  gunzip "$RAW_FASHION_DIR/train-labels-idx1-ubyte.gz"
  echo "Converting fashion..."
  python2 tools/binary_converter.py "$RAW_FASHION_DIR/train-images-idx3-ubyte" "$FASHION_DIR/data.bin"
  if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm "$RAW_FASHION_DIR/train-images-idx3-ubyte"
  fi
fi

if [ ! -f "$FASHION_DIR/corpus.bin" ]; then
  python2 tools/binary_converter.py --sample "$FASHION_DIR/data.bin" "$FASHION_DIR/tmp.bin" "$FASHION_DIR/test.bin" $N_TEST 784
  python2 tools/binary_converter.py --sample "$FASHION_DIR/tmp.bin" "$FASHION_DIR/corpus.bin" "$FASHION_DIR/validation.bin" $N_TEST 784
  rm "$FASHION_DIR/tmp.bin"
fi


N_GIST=100000
GIST_DIM=960
GIST_DIR="$DATA_DIR/gist-small"

if [ ! -f "$GIST_DIR/data.bin" ]; then
  mkdir -p "$GIST_DIR"
  if [ ! -f  gist/gist_base.fvecs ]; then
    echo "Downloading GIST..."
    wget "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" -O gist.tar.gz
    echo "Extracting GIST..."
    tar xzf gist.tar.gz
  else
    echo "GIST already downloaded, using cached version..."
  fi
  echo "Converting GIST..."
  python2 tools/binary_converter.py gist/gist_base.fvecs "$GIST_DIR/data.bin"
  if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm -r gist
    rm gist.tar.gz
  fi
fi

if [ ! -f "$GIST_DIR/corpus.bin" ]; then
  python2 tools/binary_converter.py --sample "$GIST_DIR/data.bin" "$GIST_DIR/tmp.bin" "$GIST_DIR/test.bin" $N_TEST $GIST_DIM
  python2 tools/binary_converter.py --sample "$GIST_DIR/tmp.bin" "$GIST_DIR/tmp2.bin" "$GIST_DIR/validation.bin" $N_TEST $GIST_DIM
  python2 tools/binary_converter.py --sample "$GIST_DIR/tmp2.bin" "$GIST_DIR/tmp3.bin" "$GIST_DIR/corpus.bin" $N_GIST $GIST_DIM
  rm $GIST_DIR/tmp.bin
  rm $GIST_DIR/tmp2.bin
  rm $GIST_DIR/tmp3.bin
fi


STL_N=100000
STL_DIM=9216
STL_DIR="$DATA_DIR/stl10"

if [ ! -f "$STL_DIR/data.bin" ]; then
    mkdir -p "$STL_DIR"
    if [ ! -f stl10_binary.tar.gz ]; then
      echo "Downloading STL-10..."
      wget "http://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz" -O stl10_binary.tar.gz
    else
      echo "STL-10 already downloaded, using cached version..."
    fi

    echo "Extracting STL-10..."
    tar xzf stl10_binary.tar.gz

    echo "Converting STL-10..."
    python2 tools/binary_converter.py stl10_binary/unlabeled_X.bin "$STL_DIR/data.bin"
    rm -r stl10_binary
    if [ "$REMOVE_DOWNLOADED" = true ]; then
      rm stl10_binary.tar.gz
    fi
fi

if [ ! -f "$STL_DIR/corpus.bin" ]; then
  python2 tools/binary_converter.py --sample "$STL_DIR/data.bin" "$STL_DIR/tmp.bin" "$STL_DIR/test.bin" $N_TEST $STL_DIM
  python2 tools/binary_converter.py --sample "$STL_DIR/tmp.bin" "$STL_DIR/corpus.bin" "$STL_DIR/validation.bin" $N_TEST $STL_DIM
  rm "$STL_DIR/tmp.bin"
fi


TREVI_DIR="$DATA_DIR/trevi"

if [ ! -f "$TREVI_DIR/corpus.bin" ]; then
  mkdir -p "$TREVI_DIR"
  if [ ! -f trevi.zip ]; then
    echo "Downloading Trevi..."
    wget "http://phototour.cs.washington.edu/patches/trevi.zip" -O trevi.zip
    echo "Extracting Trevi..."
  fi
  mkdir patches
  unzip -q trevi.zip -d patches
  echo "Converting Trevi..."
  module --ignore-cache load Python/3.6.3-foss-2017b
  python tools/binary_converter.py patches/ "$TREVI_DIR/data.bin"
  python2 tools/binary_converter.py --sample "$TREVI_DIR/data.bin" "$TREVI_DIR/tmp.bin" "$TREVI_DIR/test.bin" $N_TEST 4096
  python2 tools/binary_converter.py --sample "$TREVI_DIR/tmp.bin" "$TREVI_DIR/corpus.bin" "$TREVI_DIR/validation.bin" $N_TEST 4096
  rm -r patches
  if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm trevi.zip
  fi
fi


SIFT_N=100000
SIFT_TRAIN_N=10000000
SIFT_DIM=128
SIFT_DIR="$DATA_DIR/sift"

if [ ! -f "$SIFT_DIR/corpus.bin" ]; then
  mkdir -p "$SIFT_DIR"
  if [ ! -f  bigann_learn.bvecs ]; then
    if [ ! -f  bigann_learn.bvecs.gz ]; then
      echo "Downloading SIFT..."
      wget "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz" -O bigann_learn.bvecs.gz
    fi
    echo "Extracting SIFT..."
    gunzip bigann_learn.bvecs.gz
  else
    echo "SIFT already downloaded, using cached version..."
  fi
  echo "Converting SIFT..."
  python2 tools/binary_converter.py bigann_learn.bvecs "$SIFT_DIR/data.bin"
  python2 tools/binary_converter.py --sample "$SIFT_DIR/data.bin" "$SIFT_DIR/tmp.bin" "$SIFT_DIR/corpus.bin" "$SIFT_N" "$SIFT_DIM"
  python2 tools/binary_converter.py --add_sample "$SIFT_DIR/tmp.bin" "$SIFT_DIR/train.bin" "$SIFT_TRAIN_N" "$SIFT_DIM" "$SIFT_DIR/corpus.bin" "$SIFT_N"
  rm "$SIFT_DIR/data.bin"
  rm "$SIFT_DIR/tmp.bin"
  rm bigann_learn.bvecs
  if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm bigann_learn.bvecs.gz
  fi
fi

if [ ! -f "$SIFT_DIR/test.bin" ]; then
  mkdir -p "$SIFT_DIR"
  if [ ! -f  bigann_query.bvecs ]; then
    if [ ! -f  bigann_query.bvecs.gz ]; then
      echo "Downloading SIFT query set..."
      wget "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz" -O bigann_query.bvecs.gz
    fi
    echo "Extracting SIFT query set..."
    gunzip bigann_query.bvecs.gz
  else
    echo "SIFT query set already downloaded, using cached version..."
  fi
  echo "Converting SIFT query set..."
  python2 tools/binary_converter.py bigann_query.bvecs "$SIFT_DIR/data_test.bin"
  python2 tools/binary_converter.py --sample "$SIFT_DIR/data_test.bin" "$SIFT_DIR/tmp.bin" "$SIFT_DIR/test.bin" "$N_TEST" "$SIFT_DIM"
  python2 tools/binary_converter.py --sample "$SIFT_DIR/data_test.bin" "$SIFT_DIR/tmp2.bin" "$SIFT_DIR/validation.bin" "$N_TEST" "$SIFT_DIM"
  rm "$SIFT_DIR/tmp.bin"
  rm "$SIFT_DIR/tmp2.bin"
  rm "$SIFT_DIR/data_test.bin"
  rm bigann_query.bvecs
  if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm bigann_query.bvecs.gz
  fi
fi


DIM=784

for DATA_SET in mnist fashion; do
  N_CORPUS=10000
  N_TRAIN=10000
  N_VALIDATION=1000
  N_TEST=1000

  ORIGINAL_DATA_DIR="${DATA_DIR}/${DATA_SET}"
  SUBSET_DIR="${DATA_DIR}/${DATA_SET}_train${N_TRAIN}"

  if [ -f "${SUBSET_DIR}/validation.bin" ]; then
    exit
  fi

  echo "Splitting ${DATA_SET}..."

  if [ "$DATA_SET" = "mnist" ]; then
    LABELS_FILE="$RAW_DATA_DIR/mnist/train-labels-idx1-ubyte"
    CLASS1=1
    CLASS2=4
  else
    LABELS_FILE="$RAW_DATA_DIR/fashion/train-labels-idx1-ubyte"
    CLASS1=7
    CLASS2=9
  fi

  N_NOTRAIN=$((N_TRAIN + N_VALIDATION + N_TEST))
  N_NOTRAIN2=$((N_VALIDATION + N_TEST))

  if [ ! -f "${ORIGINAL_DATA_DIR}/data.bin" ]; then
    echo "Error: Original data file ${ORIGINAL_DATA_DIR}/data.bin does not exist." 1>&2
    exit
  fi

  mkdir -p "$SUBSET_DIR"
  # echo "$ORIGINAL_DATA_DIR/data.bin" "$SUBSET_DIR/train_all.bin" "$SUBSET_DIR/notrain.bin" "$DIM" "$LABELS_FILE" "$CLASS1" "$CLASS2"
  python2 tools/binary_converter2.py --split_train "$ORIGINAL_DATA_DIR/data.bin" "$SUBSET_DIR/train_all.bin" "$SUBSET_DIR/notrain.bin" "$DIM" "$LABELS_FILE" "$CLASS1" "$CLASS2"
  python2 tools/binary_converter2.py --sample "$SUBSET_DIR/train_all.bin" "$SUBSET_DIR/tmp.bin" "$SUBSET_DIR/corpus.bin" "$N_CORPUS" "$DIM"
  python2 tools/binary_converter2.py --sample "$SUBSET_DIR/notrain.bin" "$SUBSET_DIR/train.bin" "$SUBSET_DIR/notrain2.bin" "$N_NOTRAIN2" "$DIM" "$N_TRAIN"
  python2 tools/binary_converter2.py --sample "$SUBSET_DIR/notrain2.bin" "$SUBSET_DIR/validation.bin" "$SUBSET_DIR/test.bin" "$N_TEST" "$DIM"
  rm "$SUBSET_DIR/train_all.bin"
  rm "$SUBSET_DIR/tmp.bin"
  rm "$SUBSET_DIR/notrain.bin"
  rm "$SUBSET_DIR/notrain2.bin"
done


for DATA_SET in mnist fashion; do
  N_TRAIN=8000
  N_TRAIN2=8000
  N_VALIDATION=1000
  N_TEST=1000

  N_TRAIN_CHOSEN=1600
  N_TRAIN_BASE=$((N_TRAIN - N_TRAIN_CHOSEN))

  ORIGINAL_DATA_DIR="${DATA_DIR}/${DATA_SET}"
  SUBSET_DIR="${DATA_DIR}/${DATA_SET}_train${N_TRAIN2}"

  if [ -f "${SUBSET_DIR}/validation.bin" ]; then
    exit
  fi

  echo "Splitting ${DATA_SET}..."

  if [ "$DATA_SET" = "mnist" ]; then
    LABELS_FILE="$RAW_DATA_DIR/mnist/train-labels-idx1-ubyte"
    CLASS1=1
    CLASS2=4
  else
    LABELS_FILE="$RAW_DATA_DIR/fashion/train-labels-idx1-ubyte"
    CLASS1=7
    CLASS2=9
  fi

  if [ ! -f "${ORIGINAL_DATA_DIR}/corpus.bin" ]; then
    echo "Data set ${DATA_SET} not yet downloaded or converted to binary."
    exit
  fi

  N_NOTRAIN=$((N_TRAIN2 + N_VALIDATION + N_TEST))
  N_NOTRAIN2=$((N_VALIDATION + N_TEST))

  if [ ! -f "${ORIGINAL_DATA_DIR}/data.bin" ]; then
    echo "Error: Original data file ${ORIGINAL_DATA_DIR}/data.bin does not exist." 1>&2
    exit
  fi

  mkdir -p "$SUBSET_DIR"
  python2 tools/binary_converter2.py --split_train "$ORIGINAL_DATA_DIR/data.bin" "$SUBSET_DIR/base_class.bin" "$SUBSET_DIR/chosen_class.bin" "$DIM" "$LABELS_FILE" "$CLASS1" "$CLASS2"
  python2 tools/binary_converter2.py --sample "$SUBSET_DIR/chosen_class.bin" "$SUBSET_DIR/notrain2.bin" "$SUBSET_DIR/train.bin" "$N_TRAIN2" "$DIM"
  python2 tools/binary_converter2.py --sample "$SUBSET_DIR/notrain2.bin" "$SUBSET_DIR/novalidation.bin" "$SUBSET_DIR/validation.bin" "$N_VALIDATION" "$DIM"
  python2 tools/binary_converter2.py --sample "$SUBSET_DIR/novalidation.bin" "$SUBSET_DIR/rest.bin" "$SUBSET_DIR/test.bin" "$N_TEST" "$DIM"
  python2 tools/binary_converter2.py --sample "$SUBSET_DIR/base_class.bin" "$SUBSET_DIR/scrap.bin" "$SUBSET_DIR/train_base.bin" "$N_TRAIN_BASE" "$DIM"
  python2 tools/binary_converter2.py --sample "$SUBSET_DIR/rest.bin" "$SUBSET_DIR/scrap2.bin" "$SUBSET_DIR/train_chosen.bin" "$N_TRAIN_CHOSEN" "$DIM"
  python2 tools/binary_converter2.py --combine "$SUBSET_DIR/train_base.bin" "$SUBSET_DIR/train_chosen.bin" "$SUBSET_DIR/corpus.bin" "$N_TRAIN_BASE" "$N_TRAIN_CHOSEN" "$DIM"

  rm "$SUBSET_DIR/base_class.bin"
  rm "$SUBSET_DIR/chosen_class.bin"
  rm "$SUBSET_DIR/notrain2.bin"
  rm "$SUBSET_DIR/novalidation.bin"
  rm "$SUBSET_DIR/rest.bin"
  rm "$SUBSET_DIR/scrap.bin"
  rm "$SUBSET_DIR/scrap2.bin"
  rm "$SUBSET_DIR/train_base.bin"
  rm "$SUBSET_DIR/train_chosen.bin"
done
