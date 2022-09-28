#!/usr/bin/env python3

import agd.LinearParallel as lp
import numpy as np
from agd import Domain

import ma

stencil = ma.StencilForConditioning(15)
assert stencil.V3.shape[2] == 6


def A_func(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B_func(x, r, p):
    return np.ones(x.shape[1:])


def psi_func(x):
    return np.zeros(x.shape[1:])


n = 128

x = np.stack(
    np.meshgrid(
        np.linspace(-1, 1, n + 1),
        np.linspace(-1, 1, n + 1),
        indexing="ij",
    )
)
domain = Domain.Complement(Domain.Ball(), Domain.Box())

u = lp.dot_VV(x, x) - 2

u, niter = ma.NewtonRootDir(u, x, domain, A_func, B_func, psi_func, stencil)

ma.SaveResults(
    "nonconvex_dir.pickle",
    {
        "niter": niter,
    },
)
