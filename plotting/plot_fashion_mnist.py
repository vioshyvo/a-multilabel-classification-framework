#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from pytools.utils import plot_recalls, scatter_csv

algos =  ['rf-class', 'rf-class-corpus']
algonames = {'rf-class' : 'RF-CLASS', 'rf-class-corpus' : 'RF-CLASS (corpus)'}
# colors = ['#78290F', '#004E64', '#573cb5', '#dd531a', '#0c97e4']
# scatter_colors = ['#15616D', '#FF7D00']
colors = ['#3182bd', '#31a354', '#e6550d', '#756bb1', '#de2d26']
scatter_colors = ['#e6550d', '#3182bd']
datasets = ['fashion_train8000', 'mnist_train8000']
datanames = {'fashion_train8000' : 'Fashion-8000', 'mnist_train8000' : 'MNIST-8000'}
outdir = 'fig'

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = None

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
markers = ['>', 'v', 'd', '^', 'o', 'p', 'h', '<', '1', '2', '3', '4', '*', 'x', 'X', 'D', 's', 'P']
plt.figure(figsize=(12, 8))

def main(respath='article_results', k=10):
    n_sample=1000
    outfname = 'fashion-mnist-train-k' + str(k) + '.png'

    for j, dataset in enumerate(datasets):
        plt.subplot(2, 2, j + 1)
        filepath = os.path.join(respath, dataset)
        xlim = [0.5, 1]
        ymin = 100
        ymax = -100
        total = 0

        print('k = ', str(k))
        for i, algo in enumerate(algos):
            fname = os.path.join(filepath, algo + '.txt')
            if not os.path.isfile(fname):
                continue
            df = pd.read_csv(fname, delim_whitespace = True)

            algoname = algonames.get(algo, algo)
            x, y = plot_recalls(df[df['k'] == k], markers[total], colors[total], algoname)
            total += 1

            ymin = min(ymin, min(y[x > xlim[0]]))
            ymax = max(ymax, max(y[x > xlim[0]]))

        plt.xlabel('recall', fontsize=14)
        if j == 0:
            plt.ylabel('query time (s)', fontsize=14, labelpad=10)
        plt.yscale('log')
        if j == 0:
            plt.legend(loc='upper center', bbox_to_anchor=(1.0, 1.2), ncol=5, fontsize=14)
        title = datanames[dataset]
        plt.xlim(xlim)
        plt.ylim(ymin - 0.001, ymax + 0.005)
        plt.title(title, fontsize=18, y=0.85)

    for j, dataset in enumerate(datasets):
        ax = plt.subplot(2, 2, j + 3, aspect='equal')
        filepath = os.path.join(respath, dataset)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fname_tsne = os.path.join(respath, dataset, 'tsne.tsv')
        scatter_csv(fname_tsne, n_sample, scatter_colors=scatter_colors)
        ax.margins(x=0.39)


    plt.subplots_adjust(wspace=0.12)
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
