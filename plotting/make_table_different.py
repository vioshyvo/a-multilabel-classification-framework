#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import os


algos = ['rf-class', 'rf-class-corpus']
algonames = {'rf-class' : 'RF-CLASS (query)', 'rf-class-corpus': 'RF-CLASS (corpus)'}
datasets = ['random2_sd1', 'random2_sd2_5', 'random2_sd5', 'fashion_train8000', 'mnist_train8000']
datanames = {'random2_sd1' : 'Gaussian, $\sigma=1$',
             'random2_sd2_5' : 'Gaussian, $\sigma=2.5$',
             'random2_sd5' : 'Gaussian, $\sigma=5$',
             'fashion_train8000' : 'Fashion8000',
             'mnist_train8000' : 'MNIST8000'}
min_recalls = [80, 90, 95]

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = None


def main(respath='article_results', k=10):
    preamble = r'''\begin{table}[!hbtp]
\begin{center}
\caption{Query time (seconds / 1000 queries)}
\label{table:different}
\begin{tabular}{''' + ('l ' * (2 + len(algos))) + r'''}
\toprule'''

    top_row = 'data set & R (\\%) & ' + ' & '.join(algonames.get(algo, algo) for algo in algos) + r' \\'

    postamble = r'''\bottomrule
\end{tabular}
\end{center}
\end{table}'''

    print(preamble)
    print(top_row)
    print(r'\midrule')

    for j, dataset in enumerate(datasets):
        for recall in min_recalls:
            minrec = recall / 100
            print(datanames.get(dataset, dataset) + ' & %d & ' % recall, end='')
            filepath = os.path.join(respath, dataset)
            times = []
            for i, algo in enumerate(algos):
                fname = os.path.join(filepath, algo + '.txt')
                if not os.path.isfile(fname):
                    continue
                df = pd.read_csv(fname, delim_whitespace = True)

                df = df[df['k'] == k]
                df = df.sort_values('query_time')
                time = 100 if all(df['recall'] < minrec) else df[df['recall'] >= minrec].iloc[0]['query_time']
                times.append(time)
            mintime = min(times)
            strtimes = ['-' if t == 100 else '%.3f' % t  if t != mintime else r'{\bf %.3f}' % t for t in times]
            print(' & '.join(strtimes) + r' \\')
        print('\midrule')

    print(postamble)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        print('Usage:', sys.argv[0], '<result_path=../article_results> <k=10>')
        sys.exit(-1)
