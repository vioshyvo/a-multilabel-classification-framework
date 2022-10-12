#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import random


def pareto_frontier(df, xname, yname):
    df = df.sort_values(yname)
    df_out = df.iloc[[0]]
    max_rec = df_out[xname]

    for i in range(1,len(df)):
        row = df.iloc[i,:]
        if (row[xname] > max_rec).any():
            df_out = df_out.append(row)
            max_rec = row[xname]

    return df_out


def plot_recalls(df, marker, color, algo):
    print(algo, ':')
    df = pareto_frontier(df, 'recall', 'query_time')
    df = df[df['recall'] <= 0.99]
    del df['var_recall']
    del df['k']
    if 'val_recall' in df.columns:
        del df['val_recall']
        del df['val_var_recall']
        del df['val_query_time']
    if 'r0' in df.columns:
        del df['r0']
    if 'r1' in df.columns:
        del df['r1']
    if 'r2' in df.columns:
        del df['r2']
    if 'r3' in df.columns:
        del df['r3']
    if 'r4' in df.columns:
        del df['r4']
    if 'r5' in df.columns:
        del df['r5']
    if 'r6' in df.columns:
        del df['r6']
    if 'r7' in df.columns:
        del df['r7']
    if 'r8' in df.columns:
        del df['r8']
    if 'r9' in df.columns:
        del df['r9']
    if 'r10' in df.columns:
        del df['r10']
    df.insert(0, 'recall', df.pop('recall'))
    df.insert(1, 'query_time', df.pop('query_time'))
    df.index = [''] * len(df)
    print(df)
    print()
    x = df['recall']
    y = df['query_time']
    plt.plot(x, y, linewidth=2.0, label=algo, linestyle='-', marker=marker, markersize=6, color=color)
    return x, y

def plot_scatter(ax, datapath, colors, data = 2, n_corpus = 1000, n_train = 1000, s_corpus = 10, s_train = 15, alpha = 0.8):
    corpus_file = os.path.join(datapath, 'corpus.csv')
    train_file = os.path.join(datapath, 'train.csv')
    df_corpus = pd.read_csv(corpus_file, header=None)
    df_train = pd.read_csv(train_file, header=None)
    ax.scatter(df_corpus.iloc[0:n_corpus,0], df_corpus.iloc[0:n_corpus,1], s=s_corpus, c=colors[-1], alpha=alpha)
    if data == 2:
        ax.scatter(df_train.iloc[0:n_train,0], df_train.iloc[0:n_train,1], s=s_train, alpha=alpha, c=colors[0])
    else:
        n_clusters = 5
        n = int(n_train / n_clusters)
        nx = int(len(df_train) / n_clusters)

        for i in [4, 0, 1, 2, 3]:
            idx = np.arange(i * nx, i * nx + n)
            ax.scatter(df_train.iloc[idx,0], df_train.iloc[idx,1], s=s_train, c=colors[i], alpha=alpha)

def precision(df, n_trees, depth, k=10, k_build=50):
    if 'k_build' in df.columns:
        df = df[df['k_build'] == k_build]
    df2 = df[df['n_trees'] == n_trees]
    df3 = df2[df2['depth'] == depth]
    df4 = df3[df3['k'] == k]
    df6 = df4[['v', 'k', 'recall', 'cs_size', 'query_time']]

#    df0 = pd.DataFrame({'v' : [0],
#                        'k' : [k],
#                        'recall' : [1],
#                        'cs_size' : [n]})
#    df6 = df0.append(df5)

    df6['k_found'] = df6['recall'] * df6['k']
    df6 = df6[df6['k_found'] > 0]
    df6['precision'] = df6['k_found'] / df6['cs_size']
    return df6

def sample(images, labels, n_sample):
    n = images.shape[0]
    idx = np.arange(n)
    idx_corpus = idx[labels == 0]
    idx_train = idx[labels == 1]
    pos_corpus = random.sample(set(idx_corpus), n_sample)
    pos_train = random.sample(set(idx_train), n_sample)

    out_images = np.concatenate((images[pos_corpus,:], images[pos_train,:]))
    out_labels = np.concatenate((np.zeros(n_sample), np.ones(n_sample)))
    return out_images, out_labels

def scatter_csv(fname_tsne, n_sample=1000, s_corpus=10, s_train=15, alpha=0.8,
                scatter_colors = ['#45a2a8', '#cf7e15']):
    df_tsne = pd.read_csv(fname_tsne, sep = '\t')
    images, labels = sample(df_tsne.iloc[:,0:2].values, df_tsne.iloc[:,2], n_sample)
    labels = np.flip(labels)
    images = np.flip(images, axis=0)
    col = np.where(labels == 1, scatter_colors[-1], scatter_colors[0])
    s = np.where(labels == 1, s_train, s_corpus)
    plt.scatter(images[:,0], images[:,1], c=col, s=s, alpha=alpha)
    
def generate_lookup_data(filepath, treetype):
    infile = os.path.join(filepath, treetype + '.txt')
    outfile = os.path.join(filepath, treetype + '-lookup.txt')
    df = pd.read_csv(infile, delim_whitespace=True)
    df[df['v'] == 1].to_csv(outfile, sep='\t')
    
def generate_knn_data(filepath, treetype):
    infile = os.path.join(filepath, 'rf-' + treetype + '.txt')
    outfile = os.path.join(filepath, treetype + '-knn.txt')
    df = pd.read_csv(infile, delim_whitespace=True)
    df[df['v'] == 1].to_csv(outfile, sep='\t')
