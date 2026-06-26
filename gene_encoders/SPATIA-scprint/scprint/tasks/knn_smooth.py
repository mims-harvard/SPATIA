# K-nearest neighbor smoothing for high-throughput scRNA-Seq data
# (Python 3 implementation, depends on scikit-learn.
#  The command-line version also depends on click and pandas.)

# Authors:
#   Florian Wagner <florian.wagner@nyu.edu>
#   Yun Yan <yun.yan@nyumc.org>
# Copyright (c) 2017, 2018 New York University

import hashlib
import sys
import time
from math import ceil, log

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances


def _median_normalize(X: np.ndarray):
    num_transcripts = np.sum(X, axis=0)
    X_norm = (np.median(num_transcripts) / num_transcripts) * X
    return X_norm


def _freeman_tukey_transform(X: np.ndarray):
    return np.sqrt(X) + np.sqrt(X + 1)


def _calculate_pc_scores(matrix: np.ndarray, d, seed=0):
    tmatrix = _median_normalize(matrix)
    tmatrix = _freeman_tukey_transform(tmatrix)
    pca = PCA(n_components=d, svd_solver="randomized", random_state=seed)
    t0 = time.time()
    tmatrix = pca.fit_transform(tmatrix.T).T
    t1 = time.time()
    var_explained = np.cumsum(pca.explained_variance_ratio_)[-1]
    print("\tPCA took %.1f s." % (t1 - t0))
    sys.stdout.flush()
    print(
        "\tThe fraction of variance explained by the top %d PCs is %.1f %%."
        % (d, 100 * var_explained)
    )

    return tmatrix


def _calculate_pairwise_distances(X: np.ndarray, num_jobs=1):
    D = pairwise_distances(X.T, n_jobs=num_jobs, metric="euclidean")
    return D


def knn_smoothing(
    X: np.ndarray, k: int, d: int = 10, dither: float = 0.03, seed: int = 0
) -> np.ndarray:

    np.random.seed(seed)

    if not (X.dtype == np.float64 or X.dtype == np.float32):
        raise ValueError(
            "X must contain floating point values! " "Try X = np.float64(X)."
        )

    p, n = X.shape
    num_pcs = min(p, n - 1)

    if k < 1 or k > n:
        raise ValueError("k must be between 1 and and %d." % n)
    if d < 1 or d > num_pcs:
        raise ValueError("d must be between 1 and %d." % num_pcs)

    print(
        "Performing kNN-smoothing v2.1 with k=%d, d=%d, and dither=%.3f..."
        % (k, d, dither)
    )
    sys.stdout.flush()

    t0_total = time.time()

    if k == 1:
        num_steps = 0
    else:
        num_steps = ceil(log(k) / log(2))

    S = X.copy()

    for t in range(1, num_steps + 1):
        k_step = min(pow(2, t), k)
        print("Step %d/%d: Smooth using k=%d" % (t, num_steps, k_step))
        sys.stdout.flush()

        Y = _calculate_pc_scores(S, d, seed=seed)
        if dither > 0:
            for li in range(d):
                ptp = np.ptp(Y[li, :])
                dy = (np.random.rand(Y.shape[1]) - 0.5) * ptp * dither
                Y[li, :] = Y[li, :] + dy

        t0 = time.time()
        D = _calculate_pairwise_distances(Y)
        t1 = time.time()
        print("\tCalculating pair-wise distance matrix took %.1f s." % (t1 - t0))
        sys.stdout.flush()

        t0 = time.time()
        A = np.argsort(D, axis=1, kind="mergesort")
        for j in range(X.shape[1]):
            ind = A[j, :k_step]
            S[:, j] = np.sum(X[:, ind], axis=1)

        t1 = time.time()
        print("\tCalculating the smoothed expression matrix took %.1f s." % (t1 - t0))
        sys.stdout.flush()

    t1_total = time.time()
    print("kNN-smoothing finished in %.1f s." % (t1_total - t0_total))
    sys.stdout.flush()

    return S


if __name__ == "__main__":
    import click
    import pandas as pd

    @click.command()
    @click.option("-k", type=int, help="The number of neighbors to use for smoothing.")
    @click.option(
        "-d",
        default=10,
        show_default=True,
        help="The number of principal components used to identify " "neighbors.",
    )
    @click.option(
        "--dither",
        default=0.03,
        show_default=True,
        help="The amount of dither to apply to the partially "
        "smoothed and PCA-transformed data in each step. "
        "Specified as the faction of range of the scores of "
        "each PC.",
    )
    @click.option("-f", "--fpath", help="The input UMI-count matrix.")
    @click.option("-o", "--saveto", help="The output matrix.")
    @click.option(
        "-s",
        "--seed",
        default=0,
        show_default=True,
        help="Seed for pseudo-random number generator.",
    )
    @click.option(
        "--sep",
        default="\t",
        show_default=False,
        help="Separator used in input file. The output file will "
        "use this separator as well.  [default: \\t]",
    )
    @click.option(
        "--test", is_flag=True, help="Test if results for test data are correct."
    )
    def main(k, d, dither, fpath, saveto, seed, sep, test):
        print("Loading the data...", end=" ")
        sys.stdout.flush()
        t0 = time.time()
        matrix = pd.read_csv(fpath, index_col=0, sep=sep).astype(np.float64)
        t1 = time.time()
        print("done. (Took %.1f s.)" % (t1 - t0))
        sys.stdout.flush()
        p, n = matrix.shape
        print("The expression matrix contains %d genes and %d cells." % (p, n))
        sys.stdout.flush()
        print()

        S = knn_smoothing(matrix.values, k, d=d, dither=dither, seed=seed)
        print()

        print('Writing results to "%s"...' % saveto, end=" ")
        sys.stdout.flush()
        t0 = time.time()
        matrix = pd.DataFrame(S, index=matrix.index, columns=matrix.columns)
        matrix.to_csv(saveto, sep=sep)
        t1 = time.time()
        print("done. (Took %.1f s.)" % (t1 - t0))

        if test:
            with open(saveto, "rb") as fh:
                h = str(hashlib.md5(fh.read()).hexdigest())
                if h == "c8ee70f41b141b781041075e280661ff":
                    print("Test successful!!!")
                else:
                    raise ValueError("Output not correct!")

    main()
