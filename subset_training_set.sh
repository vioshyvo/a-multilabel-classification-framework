#!/usr/bin/env bash

DIM=784

for DATA_SET in mnist fashion; do
  N_CORPUS=10000
  N_TRAIN=10000
  N_VALIDATION=1000
  N_TEST=1000

  DATA_DIR="data"
  ORIGINAL_DATA_DIR="${DATA_DIR}/${DATA_SET}"
  SUBSET_DIR="${DATA_DIR}/${DATA_SET}_train${N_TRAIN}"

  if [ -f "${SUBSET_DIR}/validation.bin" ]; then
    exit
  fi

  echo "Splitting ${DATA_SET}..."

  if [ "$DATA_SET" = "mnist" ]; then
    LABELS_FILE="raw_data/mnist/train-labels-idx1-ubyte"
    CLASS1=1
    CLASS2=4
  else
    LABELS_FILE="raw_data/fashion/train-labels-idx1-ubyte"
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
