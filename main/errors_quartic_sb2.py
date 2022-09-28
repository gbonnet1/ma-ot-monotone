#!/usr/bin/env python3

import agd.LinearParallel as lp
import numpy as np
from agd import Domain

import ma

stencil = ma.StencilForConditioning(5)
assert stencil.V3.shape[2] == 2


def u_func(x):
    return lp.dot_VV(x, x) ** 2 / 4


def du_func(x):
    return lp.dot_VV(x, x) * x


def f_func(x):
    return 3 * lp.dot_VV(x, x) ** 2


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
    return f_func(x) * g_func(du_func(x)) / g_func(p)


def F_func(x, r, p0, p1):
    p = np.maximum(-p0, -p1)
    return (
        20
        * np.where(
            np.all(p >= 0, axis=0), np.sqrt(np.sum(p**2, axis=0)), np.max(p, axis=0)
        )
        - 20
    )


N = 2 ** np.arange(3, 8)
errors_u = np.zeros(N.shape)
errors_du = np.zeros(N.shape)
errors_alpha = np.zeros(N.shape)
niters = np.zeros(N.shape, dtype=np.int64)

for i, n in enumerate(N):
    print(n)

    m = 2 * np.ceil(n / 2).astype(np.int64)
    x = np.stack(
        np.meshgrid(
            np.linspace(-m / n, m / n, m + 1),
            np.linspace(-m / n, m / n, m + 1),
            indexing="ij",
        )
    )
    domain = Domain.Ball()

    u_exact = u_func(x)
    du_exact = du_func(x)

    u = lp.dot_VV(x, x)
    u0 = 0

    u, alpha, niter = ma.NewtonRootBV2(
        u, x, domain, A_func, B_func, u0, F_func, stencil
    )

    bc = Domain.Dirichlet(domain, np.inf, x)

    du0 = bc.DiffUpwind(u, [[1, 0], [0, 1]])
    du1 = bc.DiffUpwind(u, [[-1, 0], [0, -1]])
    laplacian_u = np.sum(bc.Diff2(u, [[1, 0], [0, 1]]), axis=0)
    du_defined = np.logical_and(bc.interior, laplacian_u < np.inf)
    du = np.where(du_defined, (du0 - du1) / 2, np.nan)

    error_u = np.where(bc.interior, u - u_exact, 0)
    error_du = np.where(du_defined, np.sqrt(lp.dot_VV(du - du_exact, du - du_exact)), 0)

    errors_u[i] = np.max(np.abs(error_u))
    errors_du[i] = np.max(np.abs(error_du))
    errors_alpha[i] = np.abs(alpha)
    niters[i] = niter

ma.SaveResults(
    "errors_quartic_sb2.pickle",
    {
        "N": N,
        "errors_u": errors_u,
        "errors_du": errors_du,
        "errors_alpha": errors_alpha,
        "niters": niters,
    },
)
