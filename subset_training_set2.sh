#!/usr/bin/env bash

DIM=784

for DATA_SET in mnist fashion; do
  N_TRAIN=8000
  N_TRAIN2=8000
  N_VALIDATION=1000
  N_TEST=1000

  N_TRAIN_CHOSEN=1600
  N_TRAIN_BASE=$((N_TRAIN - N_TRAIN_CHOSEN))

  DATA_DIR="data"
  ORIGINAL_DATA_DIR="${DATA_DIR}/${DATA_SET}"
  SUBSET_DIR="${DATA_DIR}/${DATA_SET}_train${N_TRAIN2}"

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
