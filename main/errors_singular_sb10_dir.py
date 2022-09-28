#!/usr/bin/env python3

import agd.LinearParallel as lp
import numpy as np
from agd import Domain

import ma

stencil = ma.StencilForConditioning(30)
assert stencil.V3.shape[2] == 10


def u_func(x):
    return -np.sqrt(2 - lp.dot_VV(x, x))


def du_func(x):
    return x / np.sqrt(2 - lp.dot_VV(x, x))


def f_func(x):
    return 2 / (2 - lp.dot_VV(x, x)) ** 2


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


def psi_func(x):
    return u_func(x)


N = 2 ** np.arange(3, 8)
errors_u = np.zeros(N.shape)
errors_du_l1 = np.zeros(N.shape)
niters = np.zeros(N.shape, dtype=np.int64)

for i, n in enumerate(N):
    print(n)

    theta = np.pi / 3
    R_theta = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    m = 2 * np.ceil(n / 2).astype(np.int64)
    x = np.stack(
        np.meshgrid(
            np.linspace(-2 * m / n, 2 * m / n, 2 * m + 1),
            np.linspace(-2 * m / n, 2 * m / n, 2 * m + 1),
            indexing="ij",
        )
    )
    domain = Domain.AffineTransform(Domain.Box(sides=((-1, 1), (-1, 1))), mult=R_theta)

    u_exact = u_func(x)
    du_exact = du_func(x)

    u = lp.dot_VV(x, x)

    u, niter = ma.NewtonRootDir(u, x, domain, A_func, B_func, psi_func, stencil)

    bc = Domain.Dirichlet(domain, np.inf, x)

    du0 = bc.DiffUpwind(u, [[1, 0], [0, 1]])
    du1 = bc.DiffUpwind(u, [[-1, 0], [0, -1]])
    laplacian_u = np.sum(bc.Diff2(u, [[1, 0], [0, 1]]), axis=0)
    du_defined = np.logical_and(bc.interior, laplacian_u < np.inf)
    du = np.where(du_defined, (du0 - du1) / 2, np.nan)

    error_u = np.where(bc.interior, u - u_exact, 0)
    error_du = np.where(du_defined, np.sqrt(lp.dot_VV(du - du_exact, du - du_exact)), 0)

    errors_u[i] = np.max(np.abs(error_u))
    errors_du_l1[i] = (2 / n) ** 2 * np.sum(np.abs(error_du))
    niters[i] = niter

ma.SaveResults(
    "errors_singular_sb10_dir.pickle",
    {
        "N": N,
        "errors_u": errors_u,
        "errors_du_l1": errors_du_l1,
        "niters": niters,
    },
)
