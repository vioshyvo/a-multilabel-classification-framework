#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from pytools.utils import plot_recalls

algos = ['rf-class']
algonames = {'rf-class': 'RF-CLASS'}
colors = ['#bd2940', '#5ca013', '#573cb5', '#dd531a', '#0c97e4']
dataset = 'sift'
datanames = {'sift' : 'SIFT'}
outdir = 'fig'

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = None

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
markers = ['>', 'v', 'd', '^', 'o', 'p', 'h', '<', '1', '2', '3', '4', '*', 'x', 'X', 'D', 's', 'P']
plt.figure(figsize=(6, 4))

def main(respath='article_results', k=10):
    outfname = 'sift-k' + str(k) + '.png'
    group_by = 'n_train'

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
        if group_by in df.columns and len(df[group_by].unique()) > 1:
            grouped = df.groupby(group_by)
            for group, df_crnt in grouped:
                print(group_by, ": ", group)
                x, y = plot_recalls(df_crnt[df_crnt['k'] == k], markers[total], colors[total], algoname + '-' + str(group))
                total += 1
                ymin = min(ymin, min(y[x > xlim[0]]) if len(y[x > xlim[0]]) > 0 else 10000)
                ymax = max(ymax, max(y[x > xlim[0]]) if len(y[x > xlim[0]]) > 0 else -10000)

        else:
            x, y = plot_recalls(df[df['k'] == k], markers[total], colors[total], algoname)
            total += 1

        ymin = min(ymin, min(y[x > xlim[0]]))
        ymax = max(ymax, max(y[x > xlim[0]]))

    plt.xlabel('recall', fontsize=14)
    plt.ylabel('query time (s)', fontsize=14)
    plt.yscale('log')
    plt.legend()
    title = dataset
    title = datanames[dataset]
    tol = ymin / 2
    plt.xlim(xlim)
    plt.ylim(ymin - tol, ymax + tol)
    plt.title(title, fontsize=18, y=0.85)

    # plt.subplots_adjust(hspace=0.20)
    # plt.subplots_adjust(wspace=0.15)
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
