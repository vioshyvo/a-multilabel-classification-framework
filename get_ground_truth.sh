#!/usr/bin/env bash

. config.sh

ALGO_DIR=bin    # path to the directory containing test code
TMP_DIR=tmp     # path to the directory where the binaries are copied and where they are ran

ALGOS=( exact exact_all_pairs ann_all_pairs )
ALGOS_TMP=( "${ALGOS[@]/%/_$1}" )

for algo in ${ALGOS[@]}; do
  cp -a "$ALGO_DIR/$algo" "$TMP_DIR/${algo}_$1"
  pushd "$TMP_DIR/${algo}_$1"
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


for DATA_SET in mnist fashion gist-small trevi stl10 sift random2_sd1 random2_sd2_5 random2_sd5; do
  DATA_DIR="$BASE_DATA_DIR/$DATA_SET"
  if [ -f "$DATA_DIR/corpus.bin" ]; then
    ./all-pairs.sh "$DATA_SET" $1
  else
    echo "File $DATA_DIR/corpus.bin not found!"
  fi
done
