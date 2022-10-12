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
RESULT_DIR="results/$DATASET_NAME"
mkdir -p "$RESULT_DIR"

ALGOS=( annoy hnsw ivf kd pca rf-class rf-class-depth rf-kd rf-pca rf-rp rp )
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

RESULT_FILE_ANNOY="$RESULT_DIR/annoy.txt"
echo "k n_trees search_k build_time recall var_recall query_time val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_ANNOY"
for n_trees in $ANNOY_N_TREES; do
    $TMP_DIR/annoy_$2/tester "$DATA_DIR" $DIM $n_trees $ANNOY_SEARCH_K >> "$RESULT_FILE_ANNOY"
done

RESULT_FILE_HNSW="$RESULT_DIR/hnsw.txt"
echo "k m efc search build_time recall var_recall query_time val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_HNSW"
for M in $HNSW_M; do
  for EFC in $HNSW_EFC; do
    $TMP_DIR/hnsw_$2/tester "$DATA_DIR" $DIM $M $EFC $HNSW_SEARCH >> "$RESULT_FILE_HNSW"
  done
done

RESULT_FILE_IVF="$RESULT_DIR/ivf.txt"
echo "k probes k_factor build_time recall var_recall query_time val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_IVF"
$TMP_DIR/ivf_$2/tester "$DATA_DIR" $DIM $IVF_PROBES $IVF_K_FACTORS >> "$RESULT_FILE_IVF"

RESULT_FILE_RF_CLASS_DEPTH="$RESULT_DIR/rf-class-depth.txt"
echo "k n_trees depth sparsity k_build b v build_time recall var_recall query_time projection_time traversal_time vote_time exact_time cs_size n_train n_subsample label_recall val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_RF_CLASS_DEPTH"
for n_subsample in $RF_CLASS_N_SUBSAMPLE; do
  for n_train in $RF_N_TRAIN; do
    for k_build in $RF_CLASS_K_BUILD; do
      for sparsity in $RF_SPARSITY; do
        for n_trees in $RF_CLASS_N_TREES; do
          for depth in $RF_CLASS_DEPTH; do
            for b in $RF_B; do
              $TMP_DIR/rf-class-depth_$2/tester "$DATA_DIR" $DIM $n_trees $depth $sparsity $k_build $b $n_train $K_BUILD $n_subsample 100 $RF_CLASS_PROB >> "$RESULT_FILE_RF_CLASS_DEPTH"
            done
          done
        done
      done
    done
  done
done

RESULT_FILE_RF_KD="$RESULT_DIR/rf-kd.txt"
echo "k n_trees depth sparsity k_build b v build_time recall var_recall query_time projection_time traversal_time vote_time exact_time cs_size n_train val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_RF_KD"
for n_train in $RF_N_TRAIN; do
  for k_build in $RF_K_BUILD; do
    for sparsity in $RF_SPARSITY; do
      for n_trees in $RF_N_TREES; do
        for depth in $RF_DEPTH; do
          for b in $RF_B; do
            $TMP_DIR/rf-kd_$2/tester "$DATA_DIR" $DIM $n_trees $depth $sparsity $k_build $b $n_train $K_BUILD $RF_VOTES >> "$RESULT_FILE_RF_KD"
          done
        done
      done
    done
  done
done

RESULT_FILE_RF_PCA="$RESULT_DIR/rf-pca.txt"
echo "k n_trees depth sparsity k_build b v build_time recall var_recall query_time projection_time traversal_time vote_time exact_time cs_size n_train val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_RF_PCA"
for n_train in $RF_N_TRAIN; do
  for k_build in $RF_K_BUILD; do
    for sparsity in $RF_SPARSITY; do
      for n_trees in $RF_N_TREES; do
        for depth in $RF_DEPTH; do
          for b in $RF_B; do
            $TMP_DIR/rf-pca_$2/tester "$DATA_DIR" $DIM $n_trees $depth $sparsity $k_build $b $n_train $K_BUILD $RF_VOTES >> "$RESULT_FILE_RF_PCA"
          done
        done
      done
    done
  done
done

RESULT_FILE_RF_RP="$RESULT_DIR/rf-rp.txt"
echo "k n_trees depth sparsity k_build b v build_time recall var_recall query_time projection_time traversal_time vote_time exact_time cs_size n_train val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_RF_RP"
for n_train in $RF_N_TRAIN; do
  for k_build in $RF_K_BUILD; do
    for sparsity in $RF_SPARSITY; do
      for n_trees in $RF_N_TREES; do
        for depth in $RF_DEPTH; do
          for b in $RF_B; do
            $TMP_DIR/rf-rp_$2/tester "$DATA_DIR" $DIM $n_trees $depth $sparsity $k_build $b $n_train $K_BUILD $RF_VOTES >> "$RESULT_FILE_RF_RP"
          done
        done
      done
    done
  done
done

RESULT_FILE_KD="$RESULT_DIR/kd.txt"
echo "k n_trees depth sparsity v build_time recall var_recall query_time cs_size val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_KD"
for sparsity in $VOTING_SPARSITY; do
  for n_trees in $VOTING_N_TREES; do
    for depth in $VOTING_DEPTH; do
      $TMP_DIR/kd_$2/tester "$DATA_DIR" $DIM $n_trees $depth $sparsity $VOTING_VOTES >> "$RESULT_FILE_KD"
    done
  done
done

RESULT_FILE_PCA="$RESULT_DIR/pca.txt"
echo "k n_trees depth sparsity v build_time recall var_recall query_time cs_size val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_PCA"
for sparsity in $VOTING_SPARSITY; do
  for n_trees in $VOTING_N_TREES; do
    for depth in $VOTING_DEPTH; do
      $TMP_DIR/pca_$2/tester "$DATA_DIR" $DIM $n_trees $depth $sparsity $VOTING_VOTES >> "$RESULT_FILE_PCA"
    done
  done
done

RESULT_FILE_RP="$RESULT_DIR/rp.txt"
echo "k n_trees depth sparsity v build_time recall var_recall query_time projection_time traversal_time vote_time exact_time cs_size val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_RP"
for sparsity in $VOTING_SPARSITY; do
  for n_trees in $VOTING_N_TREES; do
    for depth in $VOTING_DEPTH; do
      $TMP_DIR/rp_$2/tester "$DATA_DIR" $DIM $n_trees $depth $sparsity $VOTING_VOTES >> "$RESULT_FILE_RP"
    done
  done
done

RESULT_FILE_RF_CLASS="$RESULT_DIR/rf-class.txt"
echo "k n_trees depth sparsity k_build b v build_time recall var_recall query_time projection_time traversal_time vote_time exact_time cs_size n_train val_recall val_var_recall val_query_time r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10" > "$RESULT_FILE_RF_CLASS"
for n_train in $RF_N_TRAIN; do
  for k_build in $RF_CLASS_K_BUILD; do
    for sparsity in $RF_SPARSITY; do
      for n_trees in $RF_CLASS_N_TREES; do
        for depth in $RF_CLASS_DEPTH; do
          for b in $RF_B; do
            $TMP_DIR/rf-class_$2/tester "$DATA_DIR" $DIM $n_trees $depth $sparsity $k_build $b $n_train $K_BUILD $RF_CLASS_VOTES >> "$RESULT_FILE_RF_CLASS"
          done
        done
      done
    done
  done
done
