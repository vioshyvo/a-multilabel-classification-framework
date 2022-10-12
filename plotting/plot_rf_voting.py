#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from pytools.utils import pareto_frontier, generate_lookup_data

def plot_recalls2(df, linewidth, linestyle, color, algo):
    print(algo, ':')
    df = pareto_frontier(df, 'recall', 'query_time')
    df = df[df['recall'] <= 0.99]
    del df['var_recall']
    del df['k']
    if 'val_recall' in df.columns:
        del df['val_recall']
        del df['val_var_recall']
        del df['val_query_time']
    for i in range(0,10):
        colname = 'r' + str(i)
        if colname in df.columns:
            del df[colname]
    df.insert(0, 'recall', df.pop('recall'))
    df.insert(1, 'query_time', df.pop('query_time'))
    df.index = [''] * len(df)
    print(df)
    print()
    x = df['recall']
    y = df['query_time']
    plt.plot(x, y, label=algo, linewidth=linewidth, linestyle=linestyle, color=color)
    return x, y



algos = ['rp', 'rp-lookup', 'kd', 'kd-lookup', 'pca', 'pca-lookup']
algonames = {'rp': 'RP-voting', 'rf-rp': 'RF-RP', 'rf-class-depth': 'RF-CLASS',
             'rf-kd' : 'RF-KD', 'rf-pca' : 'RF-PCA', 'kd' : 'KD-voting', 
             'pca' : 'PCA-voting', 'rp-lookup' : 'RP-lookup',
             'kd-lookup' : 'KD-lookup', 'pca-lookup' : 'PCA-lookup'}
colors = ['#3182bd', '#9ecae1', '#31a354', '#a1d99b', '#e6550d', '#fdae6b', '#756bb1']
linestyles = ['dashed', 'dashed', 'dashdot', 'dashdot', (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), 'solid']
linewidths = [3, 3, 3, 3, 3, 3, 3]
datasets = ['fashion', 'gist-small', 'stl10', 'trevi']
datanames = {'mnist' : 'MNIST', 'fashion' : 'Fashion', 'gist-small' : 'GIST',
             'stl10' : 'STL-10', 'trevi': 'Trevi'}
outdir = 'fig'

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = None

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.figure(figsize=(14, 10))

def main(respath='article_results', k=10):
    outfname = 'annc-voting-k' + str(k) + '.pdf'

    for j, dataset in enumerate(datasets):
        plt.subplot(2, 2, j + 1)
        filepath = os.path.join(respath, dataset)
        xlim = [0, 1] if 'random' in dataset else [0.5, 1]
        ymins = []
        ymaxs = []

        print('k = ', str(k))
        for i, algo in enumerate(algos):
            fname = os.path.join(filepath, algo + '.txt')
            if not os.path.isfile(fname):
                treetype, search_method = algo.split('-') 
                if search_method == 'lookup':
                    generate_lookup_data(filepath, treetype)
                else:
                    continue
            df = pd.read_csv(fname, delim_whitespace = True)

            algoname = algonames.get(algo, algo)
            x, y = plot_recalls2(df[df['k'] == k], linewidths[i], linestyles[i],
                                 colors[i], algoname)

            ymins.append(min(y[x > xlim[0]]))
            ymaxs.append(max(y[x > xlim[0]]))

        ymin = min(ymins)
        ymaxs.sort()
        ymax = ymaxs[-2]

        if j == 2 or j == 3:
            plt.xlabel('recall', fontsize=18)
        if j % 2 == 0:
            plt.ylabel('query time (s)', fontsize=18)
        plt.yscale('log')
        if j == 0:
            leg = plt.legend(loc='upper center', bbox_to_anchor=(1, 1.2), ncol=7, fontsize=14)
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)
        title = datanames[dataset]
        plt.xlim(xlim)
        plt.ylim(ymin, ymax)
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
