#!/usr/bin/env python3


# This file demonstrates how to use the code from the ma.py file
# in order to solve an optimal transport problem.

# See the paper for a detailed description of the numerical method:

# G. Bonnet and J.‑M. Mirebeau.
# Monotone discretization of the Monge-Ampère equation of optimal transport.
# ESAIM: Mathematical Modelling and Numerical Analysis, 56(3):815–865, 2022.
# https://www.esaim-m2an.org/articles/m2an/abs/2022/03/m2an210105/m2an210105.html


# We start by importing some dependencies.

import agd.LinearParallel as lp  # Some low-dimensional linear algebra routines compatible with numpy.
import matplotlib.pyplot as plt
import numpy as np
from agd import Domain  # Routines to work with several kinds of geometrical domains.

import ma  # The core of the implementation of the numerical method is in the ma.py file.

# In order to apply the numerical method,
# we have to choose a set of superbases.
# Here we choose 6 superbases defined by the following property:
# For any matrix D whose condition number is less than 15,
# one of those superbases is D-obtuse.
stencil = ma.StencilForConditioning(15)
assert stencil.V3.shape[2] == 6


# The function f describes the density of the source measure
# of the optimal transport problem.
# This density is plotted at the end of the script.
def f_func(x):
    return 4 / 3 * np.where(np.logical_and(x[0] >= 0, x[1] >= 0), 0, 1) / np.pi


# The function g describes the density of the target measure
# of the optimal transport problem.
# Here for simplicity we choose it as identically one.
def g_func(y):
    return np.ones(y.shape[1:]) / np.pi


# For optimal transport problems with the quadratic cost,
# one always has A(x, p) = 0.
# (Note: the variable r has a meaning in the framework of generated
# Jacobian equations, but it can be ingored here.)
def A_func(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


# For optimal transport problems with the quadratic cost,
# one always has B(x, p) = f(x) / g(p).
def B_func(x, r, p):
    return f_func(x) / g_func(p)


# The function F describes the target domain
# (indirectly, through the optimal transport boundary condition).
#
# With the notation of the paper,
# if p0 = (\delta_h^{e_i})_{i=1,2} and p1 = (\delta_h^{-e_i})_{i=1,2},
# then F should return S_{BV2}^h u[x],
# which is defined as \max_{|e|=1} (D_h^e u[x] - \sigma_{P(x)}(e)).
#
# Here the target domain is the unit disk and the transport cost is quadratic,
# so \sigma_{P(x)}(e) = 1 for all x.
#
# According to Remark 6.2 of the paper,
# in practice we multiply S_{BV2}^h u[x] by 20,
# which improves the convergence of the Newton method.
def F_func(x, r, p0, p1):
    p = np.maximum(-p0, -p1)
    return (
        20
        * np.where(
            np.all(p >= 0, axis=0), np.sqrt(np.sum(p**2, axis=0)), np.max(p, axis=0)
        )
        - 20
    )


# Resolution of the Cartesian grid on which we solve the scheme.
n = 128

# Definition of the Cartesian grid.
# It is assumed in the code that the grid contains the origin,
# because the value of u[0] is prescribed by the scheme.
x = np.stack(
    np.meshgrid(
        np.linspace(-1, 1, n + 1),
        np.linspace(-1, 1, n + 1),
        indexing="ij",
    )
)

# We choose the unit disk as the source domain.
# It is assumed in the code that the source domain contains the origin.
domain = Domain.Ball()

# Initialization of the Newton method.
u = lp.dot_VV(x, x)

# u0 is the prescribed value of u[0]. Here we ask that u[0] = 0.
u0 = 0

# The scheme is solved at this step.
u, alpha, niter = ma.NewtonRootBV2(u, x, domain, A_func, B_func, u0, F_func, stencil)

# For information, we print the value of alpha that we obtain in the solution.
# this value is expected to be small.
print("***", alpha, "***")

# We use finite differences
# to deduce an approximation of the optimal transport map
# from the solution to the scheme.
#
# In our setting, the transport map is simply the gradient
# of the solution to the Monge-Ampère equation.
#
# We use centered finite differences,
# so we only approximate the transport map at points that are far enough
# from the boundary of the source domain (or equivalently, at points at which
# the centered finite discretization of the Laplacian is well-defined).
bc = Domain.Dirichlet(domain, np.inf, x)
du0 = bc.DiffUpwind(u, [[1, 0], [0, 1]])
du1 = bc.DiffUpwind(u, [[-1, 0], [0, -1]])
laplacian_u = np.sum(bc.Diff2(u, [[1, 0], [0, 1]]), axis=0)
du_defined = np.logical_and(bc.interior, laplacian_u < np.inf)
du = np.where(du_defined, (du0 - du1) / 2, np.nan)

# Finally, we plot the source density that we choosed,
# the solution to the scheme, and the approximate transport map.

plt.title("Source density")
im = plt.pcolormesh(*x, np.where(bc.interior, f_func(x), np.nan))
plt.colorbar(im)
plt.axis("equal")
plt.show()

plt.title("Solution to the scheme")
im = plt.pcolormesh(*x, np.where(bc.interior, u, np.nan))
plt.colorbar(im)
plt.axis("equal")
plt.show()

plt.title("Image of the Cartesian grid by the approximate transport map")
plt.plot(
    *du[:, ::2, ::2].reshape(2, -1), "."
)  # For legibility, we display only one point out of four.
plt.axis("equal")
plt.show()
