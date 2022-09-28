#!/usr/bin/env python3

import os

import agd.FiniteDifferences as fd
import agd.LinearParallel as lp
import imageio.v2 as imageio
import numpy as np
from agd import Domain
from scipy import ndimage
from scipy.interpolate import griddata

import ma

stencil = ma.StencilForConditioning(15)
assert stencil.V3.shape[2] == 6

kappa = 2 / 3
delta_x = np.cos(np.pi / 8)
delta_y = np.cos(np.pi / 8)
radius_x = np.sqrt(1 - delta_x**2) / delta_x

raw_img = imageio.imread(
    os.path.join(os.path.dirname(__file__), "../optics/image.png"), pilmode="L"
)


def unscaled_img_func(x):
    return ndimage.map_coordinates(
        raw_img,
        (
            (radius_x - x[1]) / (2 * radius_x) * raw_img.shape[0],
            (radius_x + x[0]) / (2 * radius_x) * raw_img.shape[1],
        ),
    )


def f_func(x):
    unscaled_img = unscaled_img_func(x)
    unscaled_img = (unscaled_img / 255) ** (2.2)
    return (
        unscaled_img
        * np.sum(np.where(bc.interior, n_func(x) ** 3, 0))
        / np.sum(np.where(bc.interior, unscaled_img, 0))
        * (1 - delta_y)
        / (1 - delta_x)
    )


def n_func(x):
    return 1 / np.sqrt(1 + lp.dot_VV(x, x))


def lambda_func(x, p):
    n = n_func(x)
    tmp = lp.dot_VV(p, p) + lp.dot_VV(p, x) ** 2
    h = n**2 - (1 - kappa**2) * tmp
    return (kappa * tmp + n * np.sqrt(h)) / (n**2 + kappa**2 * tmp)


def Y_func(x, r, p):
    n = n_func(x)
    lambda_ = lambda_func(x, p)
    tmp = 1 - kappa * lambda_
    return (lambda_ * n**2 * x + tmp * p) / (lambda_ * n**2 - tmp * lp.dot_VV(p, x))


def Z_func(x, r, p):
    lambda_ = lambda_func(x, p)
    return -np.log(1 - kappa * lambda_) / kappa - r


def A_func(x, r, p):
    n = n_func(x)
    ma.spad_sum(n)
    lambda_ = lambda_func(x, p)
    ma.spad_sum(lambda_)
    return n**2 * (
        lambda_
        / (1 - kappa * lambda_)
        * (n**2 * lp.outer(x, x) - lp.identity(x.shape[1:]))
        - lp.outer(x, p)
        - lp.outer(p, x)
    ) + kappa * lp.outer(p, p)


def B_func(x, r, p):
    n = n_func(x)
    ma.spad_sum(n)
    lambda_ = lambda_func(x, p)
    ma.spad_sum(lambda_)
    tmp = 1 - kappa * lambda_
    ma.spad_sum(tmp)
    m = n * lambda_ - tmp * lp.dot_VV(p, x) / n
    ma.spad_sum(m)
    return (
        f_func(x)
        / m
        * (
            (
                n**4
                - (lambda_ + kappa - kappa * lambda_**2) * n**2 * lp.dot_VV(p, x)
            )
            / tmp**2
            - lp.dot_VV(p, p) / tmp
            + n**2 * lp.dot_VV(lp.perp(p), x) ** 2
        )
    )


def sigma_func(x, r, e):
    n = n_func(x)
    p = (1 - kappa * delta_y * n) * e - n**2 * lp.dot_VV(e, x) * x
    y = (
        np.sqrt(1 - delta_y**2)
        / (delta_y * lp.dot_VV(p, p))
        * (
            np.sqrt(
                lp.dot_VV(p, p)
                - kappa**2
                * (1 - delta_y**2)
                * n**2
                * lp.dot_VV(lp.perp(e), x) ** 2
            )
            * p
            + kappa
            * np.sqrt(1 - delta_y**2)
            * n
            * lp.dot_VV(lp.perp(e), x)
            * lp.perp(p)
        )
    )

    return (
        n * delta_y * lp.dot_VV(e, y)
        - delta_y * n**3 * (1 + lp.dot_VV(x, y)) * lp.dot_VV(e, x)
    ) / (1 - kappa * delta_y * n * (1 + lp.dot_VV(x, y)))


def F_func(x, r, p0, p1):
    t = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    e = np.multiply.outer(np.stack([np.cos(t), np.sin(t)]), np.ones(x.shape[1:]))
    x = np.expand_dims(x, 1)
    p0 = np.expand_dims(p0, 1)
    p1 = np.expand_dims(p1, 1)
    return 20 * np.max(
        np.sum(np.where(e <= 0, e * p0, -e * p1), axis=0) - sigma_func(x, r, e),
        axis=0,
    )


x = np.stack(
    np.meshgrid(
        np.linspace(-radius_x, radius_x, 121),
        np.linspace(-radius_x, radius_x, 121),
        indexing="ij",
    )
)
domain = Domain.Ball(radius=radius_x)

# Used in f_func => has to be defined here.
bc = Domain.Dirichlet(domain, np.inf, x)

target = np.flip(np.where(bc.interior, unscaled_img_func(x), 255).T, axis=0)

u = -1 / kappa * np.log(1 - kappa * n_func(x))
u = u - u.flatten()[np.argmin(lp.dot_VV(x, x))]
u0 = 0

u, _, niter = ma.NewtonRootBV2(u, x, domain, A_func, B_func, u0, F_func, stencil)

gridscale = x[0, 1, 0] - x[0, 0, 0]
p = fd.DiffCentered(np.where(bc.interior, u, np.nan), [[1, 0], [0, 1]], gridscale)

interior = np.all(np.isfinite(p), axis=0)
y = Y_func(x[:, interior], u[interior], p[:, interior])
v = Z_func(x[:, interior], u[interior], p[:, interior])

v = v - v.flatten()[np.argmin(domain.level(y))]

r_primal = np.exp(kappa * v) * n_func(y) * np.stack([y[0], y[1], np.ones(y.shape[1:])])

y3d = np.stack(
    np.meshgrid(
        np.linspace(np.min(r_primal[0]), np.max(r_primal[0]), 500),
        np.linspace(np.min(r_primal[1]), np.max(r_primal[1]), 500),
        indexing="ij",
    )
)
v3d = griddata(
    (r_primal[0], r_primal[1]),
    r_primal[2],
    (y3d[0], y3d[1]),
    method="cubic",
)

step = 5
gridscale = y3d[0, 1, 0] - y3d[0, 0, 0]
v3dx = fd.DiffCentered(v3d, (step, 0), gridscale) / step
v3dy = fd.DiffCentered(v3d, (0, step), gridscale) / step
v3dxx = fd.Diff2(v3d, (step, 0), gridscale) / step**2
v3dyy = fd.Diff2(v3d, (0, step), gridscale) / step**2
v3dxy = (
    0.25
    * (fd.Diff2(v3d, (step, step), gridscale) - fd.Diff2(v3d, (step, -step), gridscale))
    / step**2
)
v3dcurv = (v3dxx * v3dyy - v3dxy**2) / (1 + v3dx**2 + v3dy**2) ** 2

ma.SaveResults(
    "optics.pickle",
    {
        "niter": niter,
        "target": target,
        "y3d": y3d,
        "v3d": v3d,
        "v3dcurv": v3dcurv,
    },
)
