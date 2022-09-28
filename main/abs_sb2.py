#!/usr/bin/env python3

import agd.LinearParallel as lp
import numpy as np
from agd import Domain

import ma

stencil = ma.StencilForConditioning(5)
assert stencil.V3.shape[2] == 2


def A_func(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B_func(x, r, p):
    return np.zeros(x.shape[1:])


def psi_func(x):
    return np.abs(x[0] + x[1] / np.sqrt(10))


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

u, niter = ma.NewtonRootDir(u, x, domain, A_func, B_func, psi_func, stencil)

ma.SaveResults(
    "abs_sb2.pickle",
    {
        "V1": stencil.V1,
        "x": x,
        "u": u,
        "niter": niter,
    },
)
