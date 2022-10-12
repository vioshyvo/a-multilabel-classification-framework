#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from pytools.utils import pareto_frontier, generate_lookup_data, generate_knn_data

def plot_recalls2(df, linewidth, linestyle, color, algo, is_first=True, plot=True):
    print(algo, ':')
    df = pareto_frontier(df, 'recall', 'query_time')
    df = df[df['recall'] <= 0.99]
    del df['var_recall']
    del df['k']
    if 'val_recall' in df.columns:
        del df['val_recall']
        del df['val_var_recall']
        del df['val_query_time']
    for i in range(0,11):
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
    if plot:
        if is_first:
            plt.plot(x, y, label=algo, linewidth=linewidth, linestyle=linestyle, color=color)
        else:
            plt.plot(x, y, linewidth=linewidth, linestyle=linestyle, color=color)
    return x, y


algo_list = [['rf-rp', 'rp-knn', 'rp', 'rp-lookup'],
            ['rf-kd', 'kd-knn', 'kd', 'kd-lookup'],
            ['rf-pca', 'pca-knn', 'pca', 'pca-lookup']]
algonames = {'rf-rp': 'Natural classifier', 'rp-knn': r'Natural classifier, $\tau=0$',
             'rp' : 'Naive classifier (voting)',
             'rp-lookup' : r'Naive classifier, $\tau=0$ (lookup)'}
colors = ['#3182bd', '#31a354', '#e6550d', '#756bb1']
linestyles = ['solid', 'dashed', 'dashdot', (0, (3, 1, 1, 1, 1, 1))]
linewidths = [3, 3, 3, 3]
datasets = ['fashion', 'gist-small', 'stl10', 'trevi']
datanames = {'mnist' : 'MNIST', 'fashion' : 'Fashion', 'gist-small' : 'GIST',
             'stl10' : 'STL-10', 'trevi': 'Trevi'}
outdir = 'fig'

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = None

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.figure(figsize=(18, 18))

def main(respath='article_results', k=10):
    outfname = 'annc-all-k' + str(k) + '-even.pdf'

    l = 1
    for j, dataset in enumerate(datasets):
        filepath = os.path.join(respath, dataset)
        xlim = [0, 1] if 'random' in dataset else [0.5, 1]
        print('k = ', str(k))

        ymins = []
        ymaxs = []

        for u, algos in enumerate(algo_list):
            for i, algo in enumerate(algos):
                fname = os.path.join(filepath, algo + '.txt')
                if not os.path.isfile(fname):
                    treetype, search_method = algo.split('-')
                    if search_method == 'lookup':
                        generate_lookup_data(filepath, treetype)
                    elif search_method == 'knn':
                        generate_knn_data(filepath, treetype)
                    else:
                        continue
                df = pd.read_csv(fname, delim_whitespace = True)

                algoname = algonames.get(algo, algo)
                x, y = plot_recalls2(df[df['k'] == k], linewidths[i], linestyles[i],
                                     colors[i], algoname, u == 0 and j == 0, False)
                ymins.append(min(y[x > xlim[0]]))
                ymaxs.append(max(y[x > xlim[0]]))

        ymin = min(ymins)
        ymaxs.sort()
        ymax = 2 if dataset == 'fashion' else ymaxs[-1]


        for u, algos in enumerate(algo_list):
            plt.subplot(4, 3, l)
            print("l: ", l)

            for i, algo in enumerate(algos):
                fname = os.path.join(filepath, algo + '.txt')
                if not os.path.isfile(fname):
                    treetype, search_method = algo.split('-')
                    if search_method == 'lookup':
                        generate_lookup_data(filepath, treetype)
                    elif search_method == 'knn':
                        generate_knn_data(filepath, treetype)
                    else:
                        continue
                df = pd.read_csv(fname, delim_whitespace = True)

                algoname = algonames.get(algo, algo)
                x, y = plot_recalls2(df[df['k'] == k], linewidths[i], linestyles[i],
                                     colors[i], algoname, u == 0 and j == 0)
                if(i == len(algos) - 1):
                    algoname = algo.split('-')[0].upper()

            l = l + 1

            if l == 3:
                plt.xlabel('recall', fontsize=16)
            if u == 0:
                plt.ylabel('query time (s)', fontsize=16)
            plt.yscale('log')
            if j == 0 and u == 0:
                leg = plt.legend(loc='upper center', bbox_to_anchor=(1.6, 1.2), ncol=7, fontsize=14)
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)
            title = datanames[dataset] + ', ' + algoname
            plt.xlim(xlim)
            plt.ylim(ymin, ymax)
            plt.title(title, fontsize=18, y=0.85)

    plt.subplots_adjust(hspace=0.12)
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
