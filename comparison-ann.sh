#!/usr/bin/env bash

if [ ! -f "parameters/$1.sh" ]; then
  echo Invalid data set 1>&2
  exit
fi

module purge
module load GCC/5.4.0-2.26
module load OpenBLAS/0.2.18-GCC-5.4.0-2.26-LAPACK-3.6.1
module load CMake/3.6.1-foss-2016b
module load Boost/1.61.0-foss-2016b
module load GSL/2.1-foss-2016b

DATASET_NAME="$1"
. "parameters/$DATASET_NAME.sh"
. config.sh
ALGO_DIR=bin  # path to the directory containing test code
TMP_DIR=tmp   # path to the directory where the binaries are copied and where they are ran
DATA_DIR="$BASE_DATA_DIR/$DATASET_NAME"
RESULT_DIR="results/${DATASET_NAME}-ann"
mkdir -p "$RESULT_DIR"

ALGOS=( rf-class-depth )
ALGOS_TMP=( "${ALGOS[@]/%/_$2}" )

for algo in ${ALGOS[@]}; do
  cp -a "$ALGO_DIR/$algo" "$TMP_DIR/${algo}_$2"
  pushd "$TMP_DIR/${algo}_$2"
  make
  popd
done

set -e
function cleanup {
  for algo in ${ALGOS_TMP[@]}; do
    rm  -r "$TMP_DIR/$algo"
  done
}
trap cleanup EXIT


RESULT_FILE_RF_CLASS_DEPTH="$RESULT_DIR/rf-class-depth.txt"
echo "k n_trees depth sparsity k_build b v build_time recall var_recall query_time projection_time traversal_time vote_time exact_time cs_size n_train n_subsample label_recall val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_RF_CLASS_DEPTH"
for k_approximate in $RF_K_APPROXIMATE; do
  for n_subsample in $RF_CLASS_N_SUBSAMPLE; do
    for n_train in $RF_N_TRAIN; do
      for k_build in $RF_CLASS_K_BUILD; do
        for sparsity in $RF_SPARSITY; do
          for n_trees in $RF_CLASS_N_TREES; do
            for depth in $RF_CLASS_DEPTH; do
              for b in $RF_B; do
                $TMP_DIR/rf-class-depth_$2/tester "$DATA_DIR" $DIM $n_trees $depth $sparsity $k_build $b $n_train $K_BUILD $n_subsample $k_approximate $RF_CLASS_PROB >> "$RESULT_FILE_RF_CLASS_DEPTH"
              done
            done
          done
        done
      done
    done
  done
done
