#!/usr/bin/env python3

import agd.LinearParallel as lp
import numpy as np
from agd import Domain

import ma

stencil = ma.StencilForConditioning(15)
assert stencil.V3.shape[2] == 6


def f_func(x):
    return np.ones(x.shape[1:]) / np.pi


def g_func_unbalanced(y):
    y1 = np.stack([y[0] + 0.6, y[1] + 0.1])
    y2 = np.stack([y[0] - 0.6, y[1] + 0.1])
    y3 = np.stack([y[0], y[1] - 0.6])
    return (
        1 / 10
        + np.exp(-lp.dot_VV(y1, y1) / (2 * 0.1**2))
        + np.exp(-lp.dot_VV(y2, y2) / (2 * 0.1**2))
        + np.exp(-lp.dot_VV(y3, y3) / (2 * 0.1**2))
    )


g_scale_factor = 1 / ma.DiskQuad(g_func_unbalanced)


def g_func(y):
    return g_scale_factor * g_func_unbalanced(y)


def A_func(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B_func(x, r, p):
    return f_func(x) / g_func(p)


def F_func(x, r, p0, p1):
    p = np.maximum(-p0, -p1)
    return (
        20
        * np.where(
            np.all(p >= 0, axis=0), np.sqrt(np.sum(p**2, axis=0)), np.max(p, axis=0)
        )
        - 20
    )


n = 128

x = np.stack(
    np.meshgrid(
        np.linspace(-1, 1, n + 1),
        np.linspace(-1, 1, n + 1),
        indexing="ij",
    )
)
domain = Domain.Ball()

u = lp.dot_VV(x, x)
u0 = 0

u, alpha, niter = ma.NewtonRootBV2(u, x, domain, A_func, B_func, u0, F_func, stencil)

print("***", alpha, "***")

bc = Domain.Dirichlet(domain, np.inf, x)

du0 = bc.DiffUpwind(u, [[1, 0], [0, 1]])
du1 = bc.DiffUpwind(u, [[-1, 0], [0, -1]])
laplacian_u = np.sum(bc.Diff2(u, [[1, 0], [0, 1]]), axis=0)
du_defined = np.logical_and(bc.interior, laplacian_u < np.inf)
du = np.where(du_defined, (du0 - du1) / 2, np.nan)

ma.SaveResults(
    "const_to_gaussians.pickle",
    {
        "x": x,
        "f": np.where(bc.interior, f_func(x), np.nan),
        "du": du,
        "niter": niter,
    },
)
