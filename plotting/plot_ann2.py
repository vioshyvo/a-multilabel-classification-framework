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

algo = 'rf-class-depth'
algonames = {'rf-class-depth': 'RF'}
colors = ['#3182bd', '#31a354', '#e6550d', '#756bb1']
linestyles = ['solid', 'dashed', 'dashdot', (0, (3, 1, 1, 1, 1, 1))]
linewidths = [3, 3, 3, 3]
datasets = ['fashion-ann']
datanames = {'fashion-ann' : 'Fashion', 'mnist-ann' : 'MNIST'}
outdir = 'fig'

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = None

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
markers = ['>', 'v', 'd', '^', 'o', 'p', 'h', '<', '1', '2', '3', '4', '*', 'x', 'X', 'D', 's', 'P']
plt.figure(figsize=(10, 7))

def main(respath='article_results', k=10):
    outfname = 'ann-fashion-k' + str(k) + '.pdf'
    group_by = 'label_recall'
    print('k = ', str(k))
    for j, dataset in enumerate(datasets):
        xlim = [0.5, 1]
        ymin = 100
        ymax = -100
        total = 0
        filepath = os.path.join(respath, dataset)

        fname = os.path.join(filepath, algo + '.txt')
        if not os.path.isfile(fname):
            continue
        df = pd.read_csv(fname, delim_whitespace = True)

        algoname = algonames.get(algo, algo)
        if group_by in df.columns and len(df[group_by].unique()) > 1:
            grouped = df.groupby(group_by)
            for group, df_crnt in grouped:
                print("group: ", group)
                if group not in set([60, 80, 95]):
                    print(group_by, ": ", group)
                    # x, y = plot_recalls(df_crnt[df_crnt['k'] == k], markers[total], colors[total], algoname + '-' + str(group) + '%')
                    x, y = plot_recalls2(df_crnt[df_crnt['k'] == k], linewidths[total], linestyles[total],
                                 colors[total], algoname + '-' + str(group) + '%')

                    total += 1
                    ymin = min(ymin, min(y[x > xlim[0]]) if len(y[x > xlim[0]]) > 0 else 10000)
                    ymax = max(ymax, max(y[x > xlim[0]]) if len(y[x > xlim[0]]) > 0 else -10000)
            total += 1

            ymin = min(ymin, min(y[x > xlim[0]]))
            ymax = max(ymax, max(y[x > xlim[0]]))

        plt.xlabel('recall', fontsize=16)
        if j == 0:
            # plt.legend(loc='upper center', bbox_to_anchor=(1, 1.2), ncol=4, fontsize=14)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, fontsize=16)
            plt.ylabel('query time (s)', fontsize=16)
        plt.yscale('log')
        title = dataset
        title = datanames[dataset]
        tol = ymin / 2
        plt.xlim(xlim)
        plt.ylim(ymin - tol, ymax + tol)
        plt.title(title, fontsize=20, y=0.85)

    plt.subplots_adjust(hspace=0.10)
    plt.subplots_adjust(wspace=0.15)
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
