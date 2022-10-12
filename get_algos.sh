#!/usr/bin/env bash

module purge
module load GCC/5.4.0-2.26
module load OpenBLAS/0.2.18-GCC-5.4.0-2.26-LAPACK-3.6.1
module load CMake/3.6.1-foss-2016b
module load Boost/1.61.0-foss-2016b
module load GSL/2.1-foss-2016b

gcc -march=native -Q --help=target | grep march
. config.sh

pushd lib
  if [ ! -d faiss/lib ]; then
     if [ ! -d faiss ]; then
       git clone https://github.com/facebookresearch/faiss.git
     fi
     pushd faiss
     ./configure --without-cuda --prefix="$DIR/lib/faiss/lib"
      make
      make install
     popd
  fi


  if [ ! -d lz4 ]; then
     git clone https://github.com/lz4/lz4.git
     pushd lz4
       make
     popd
  fi
popd
