#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import re

from pytools.utils import plot_recalls, plot_scatter

outdir = 'fig'

algos =  ['rf-class', 'rf-class-corpus']
algonames = {'rf-class': 'RF-CLASS', 'rf-class-corpus' : 'RF-CLASS (corpus)'}
colors = ['#3182bd', '#31a354', '#e6550d', '#756bb1', '#de2d26']
scatter_colors = ['#e6550d', '#3182bd']
data = 'random2'
sds = ['1', '2_5', '5']

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = None

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
markers = ['>', 'v', 'd', '^', 'o', 'p', 'h', '<', '1', '2', '3', '4', '*', 'x', 'X', 'D', 's', 'P']
plt.figure(figsize=(16, 7))

def main(respath='article_results', k=10):
    outfname = data + '-k' + str(k) + '.png'

    for j, sd in enumerate(sds):
        ax = plt.subplot(2, 3, j + 1)
        dataset = data + '_sd' + sd
        filepath = os.path.join(respath, dataset)
        xlim = [0, 1] if 'random' in dataset else [0.5, 1]
        ymin = 100
        ymax = -100
        total = 0

        print('k = ', str(k))
        for i, algo in enumerate(algos):
            fname = os.path.join(filepath, algo + '.txt')
            if not os.path.isfile(fname):
                continue

            algoname = algonames.get(algo, algo)
            df = pd.read_csv(fname, delim_whitespace = True)
            x, y = plot_recalls(df[df['k'] == k], markers[total], colors[total], algoname)
            total += 1

            ymin = min(ymin, min(y[x > xlim[0]]))
            ymax = max(ymax, max(y[x > xlim[0]]))

            plt.xlabel('recall', fontsize=14)
            if j == 0:
                plt.ylabel('query time (s)', fontsize=14)
            plt.yscale('log')
            tol = ymin / 2
            plt.xlim(xlim)
            plt.ylim(ymin - tol, ymax + tol)

        if j == 0:
            plt.legend(loc='upper center', bbox_to_anchor=(1.6, 1.32), ncol=6, columnspacing=3, fontsize=13)

        title = '$\sigma = ' + re.sub('_', '.', sd) + '$'
        plt.title(title, fontsize=14, y=1.02)

        ins = ax.inset_axes([0.05,0.55,0.27,0.4])
        ins.set_xlim((-10.5, 10.5))
        ins.set_ylim((-10.5, 10.5))
        ins.spines['right'].set_visible(False)
        ins.spines['top'].set_visible(False)
        ins.spines['left'].set_visible(False)
        ins.spines['bottom'].set_visible(False)
        ins.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False) # labels along the bottom edge are off
        # ins.set_aspect('equal')
        plot_scatter(ins, filepath, scatter_colors)


    plt.subplots_adjust(hspace=0.2)
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
