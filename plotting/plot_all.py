#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from pytools.utils import plot_recalls


algos = ['rf-class-depth', 'rf-pca', 'hnsw', 'ivf', 'annoy']
algonames = {'rf-class-depth': 'RF-CLASS', 'rf-pca' : 'RF-PCA',
             'hnsw': 'HNSW', 'ivf': 'IVF-PQ', 'annoy' : 'ANNOY'}
colors = ['#bd2940', '#5ca013', '#573cb5', '#dd531a', '#0c97e4']
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 1)), (0, (3, 1, 1, 1))]
linewidths = [2, 2, 2, 2, 2, 2, 2]
markers = ['>', 'v', 'd', '^', 'o', 'p', 'h', '<', '1', '2', '3', '4', '*', 'x', 'X', 'D', 's', 'P']
datasets = ['mnist', 'fashion', 'gist-small', 'stl10', 'trevi']
datanames = {'mnist' : 'MNIST', 'fashion' : 'Fashion', 'gist-small' : 'GIST',
             'stl10' : 'STL-10', 'trevi': 'Trevi'}
outdir = 'fig'

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = None

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.figure(figsize=(14, 14))

def main(respath='article_results', k=10):
    outfname = 'all-k' + str(k) + '.pdf'

    for j, dataset in enumerate(datasets):
        plt.subplot(3, 2, j + 1)
        filepath = os.path.join(respath, dataset)
        xlim = [0, 1] if 'random' in dataset else [0.5, 1]
        ymin = 100
        ymax = -100

        print('k = ', str(k))
        for i, algo in enumerate(algos):
            fname = os.path.join(filepath, algo + '.txt')
            if not os.path.isfile(fname):
                continue
            df = pd.read_csv(fname, delim_whitespace = True)

            algoname = algonames.get(algo, algo)
            x, y = plot_recalls(df[df['k'] == k], markers[i], colors[i], algoname)

            tol = ymin / 2
            ymin = min(ymin, min(y[x > xlim[0]]))
            ymax = max(ymax, max(y[x > xlim[0]]))

        if j == len(datasets) - 1:
            plt.xlabel('recall', fontsize=18)
        if j % 2 == 0:
            plt.ylabel('query time (s)', fontsize=18)
        plt.yscale('log')
        if j == 0:
            plt.legend(loc='upper center', bbox_to_anchor=(1, 1.2), ncol=5, fontsize=14)
        title = datanames[dataset]
        plt.xlim(xlim)
        plt.ylim(ymin - tol, ymax + tol)
        plt.title(title, fontsize=18, y=0.85)

    plt.subplots_adjust(hspace=0.10)
    plt.subplots_adjust(wspace=0.10)
    plt.savefig(os.path.join(outdir, outfname), bbox_inches='tight')
    print('\n')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        print('Usage:', sys.argv[0], '<result_path=article_results> <k=10>')
        sys.exit(-1)
