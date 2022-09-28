#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np

import common

plt.figure(figsize=(9, 6))


def source(experiment, n):
    results = common.LoadResults(f"{experiment}_to_const.pickle")

    x_const = results["x"]
    f_const = results["f"]
    du_const = results["du"]
    niter_const = results["niter"]

    results = common.LoadResults(f"{experiment}_to_gaussians.pickle")

    x_gaussians = results["x"]
    f_gaussians = results["f"]
    du_gaussians = results["du"]
    niter_gaussians = results["niter"]

    x = x_const
    assert np.allclose(x, x_gaussians)

    f = f_const
    assert np.allclose(f, f_gaussians, equal_nan=True)

    N = du_const.shape[1]
    assert N == du_const.shape[2]
    assert N == du_gaussians.shape[1]
    assert N == du_gaussians.shape[2]
    assert N % 2 == 1

    k = 2
    M = 2 * ((N - 1) // (2 * k)) + 1
    j = (N - 1) // 2 % k

    plt.subplot(3, 5, n)
    plt.pcolormesh(*x, -f, cmap="Greys", vmax=0)
    t = np.linspace(-np.pi, np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), color="tab:red", linewidth=1)
    plt.axis("scaled")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_facecolor("k")

    plt.subplot(3, 5, n + 5)
    for i in range(M):
        plt.plot(*du_const[:, k * i + j, :], color="tab:blue", linewidth=0.2)
        plt.plot(*du_const[:, :, k * i + j], color="tab:blue", linewidth=0.2)
    t = np.linspace(-np.pi, np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), color="tab:red", linewidth=1)
    plt.axis("scaled")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"Newton iterations: {niter_const}")

    plt.subplot(3, 5, n + 10)
    for i in range(M):
        plt.plot(*du_gaussians[:, k * i + j, :], color="tab:blue", linewidth=0.2)
        plt.plot(*du_gaussians[:, :, k * i + j], color="tab:blue", linewidth=0.2)
    t = np.linspace(-np.pi, np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), color="tab:red", linewidth=1)
    plt.axis("scaled")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"Newton iterations: {niter_gaussians}")


source("const", 1)
source("ring", 2)
source("angle", 3)
source("three", 4)
source("rand", 5)

plt.subplots_adjust(
    left=0.01, bottom=0.05, right=0.99, top=0.99, wspace=0.05, hspace=0.15
)

plt.savefig(os.path.join(os.path.dirname(__file__), "out/transport_maps.png"), dpi=300)
