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

DATA_SET="$1"
. "parameters/$DATA_SET.sh"
. config.sh
TMP_DIR=tmp     # path to the directory where the binaries are copied and where they are ran
DATA_DIR="$BASE_DATA_DIR/$DATA_SET"
OUTFNAME="$DATA_DIR/exact_all_pairs_${K_BUILD}.bin"
OUTFNAME2="$DATA_DIR/exact_all_pairs2_${K_BUILD}.bin"


for k in 1 10 100; do
  if [ ! -f "$DATA_DIR/truth_$k" ]; then
    $TMP_DIR/exact_$2/tester "$DATA_DIR/corpus.bin" "$DATA_DIR/test.bin" $DIM $k > "$DATA_DIR/truth_$k"
  fi
  if [ ! -f "$DATA_DIR/validation_$k" ]; then
    $TMP_DIR/exact_$2/tester "$DATA_DIR/corpus.bin" "$DATA_DIR/validation.bin" $DIM $k > "$DATA_DIR/validation_$k"
  fi
done

if [ -f "$DATA_DIR/train.bin" ]; then
  if [ ! -f "$OUTFNAME" ]; then
    $TMP_DIR/exact_all_pairs_$2/tester "$DATA_DIR/corpus.bin" "$DATA_DIR/train.bin" $DIM $K_BUILD "$DATA_DIR" "$OUTFNAME"
  fi
  if [ ! -f "$OUTFNAME2" ]; then
    $TMP_DIR/exact_all_pairs_$2/tester "$DATA_DIR/corpus.bin" "$DATA_DIR/corpus.bin" $DIM $K_BUILD "$DATA_DIR" "$OUTFNAME2"
  fi
else
  if [ ! -f "$OUTFNAME" ]; then
    $TMP_DIR/exact_all_pairs_$2/tester "$DATA_DIR/corpus.bin" "$DATA_DIR/corpus.bin" $DIM $K_BUILD "$DATA_DIR" "$OUTFNAME"
  fi
fi


for k_approximate in $RF_K_APPROXIMATE; do
  ANNFNAME="$DATA_DIR/exact_all_pairs_${k_approximate}_${K_BUILD}.bin"
  ANNFNAME2="$DATA_DIR/exact_all_pairs2_${k_approximate}_${K_BUILD}.bin"

  if [ "$k_approximate" = 100 ]; then
    cp "$OUTFNAME" "$ANNFNAME"
  else
    if [ -f "$DATA_DIR/train.bin" ]; then
      if [ ! -f "$ANNFNAME" ]; then
        $TMP_DIR/ann_all_pairs_$2/tester "$DATA_DIR/corpus.bin" "$DATA_DIR/train.bin" $DIM $K_BUILD "$k_approximate" "$ANNFNAME"
      fi
      if [ ! -f "$ANNFNAME2" ]; then
        $TMP_DIR/ann_all_pairs_$2/tester "$DATA_DIR/corpus.bin" "$DATA_DIR/corpus.bin" $DIM $K_BUILD "$k_approximate" "$ANNFNAME2"
      fi
    else
      if [ ! -f "$ANNFNAME" ]; then
        $TMP_DIR/ann_all_pairs_$2/tester "$DATA_DIR/corpus.bin" "$DATA_DIR/corpus.bin" $DIM $K_BUILD "$k_approximate" "$ANNFNAME"
      fi
    fi
  fi
done
