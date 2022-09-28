import os
import pickle
import time
from dataclasses import dataclass

import agd.LinearParallel as lp
import numpy as np
from agd import Domain, Selling
from agd.AutomaticDifferentiation import Optimization, Sparse
from agd.AutomaticDifferentiation.misc import tocsr
from agd.AutomaticDifferentiation.Sparse import spAD
from scipy.integrate import dblquad
from scipy.sparse.linalg import spsolve


def spad_sum(a):
    if not isinstance(a, spAD):
        return

    index, inverse = np.unique(a.index, axis=-1, return_inverse=True)
    coef = np.zeros(index.shape)

    for i in range(index.shape[-1]):
        coef[..., i] = np.sum(a.coef[..., inverse == i], axis=-1)

    a.index = index
    a.coef = coef


@dataclass
class Stencil:
    V1: np.ndarray
    V2: np.ndarray
    V2_indices: np.ndarray
    V3: np.ndarray
    V3_indices: np.ndarray
    Q: np.ndarray
    w: np.ndarray
    omega0: np.ndarray
    omega1: np.ndarray
    omega2: np.ndarray


def StencilForConditioning(cond):
    V3 = Selling.SuperbasesForConditioning(cond)
    offsets = V3.reshape((2, -1))

    # Make offsets positive for the lexicographic order, inversing their sign if needed.
    offsets[:, offsets[0] < 0] *= -1
    offsets[:, np.logical_and(offsets[0] == 0, offsets[1] < 0)] *= -1

    V1, indices = np.unique(offsets, axis=1, return_inverse=True)
    V3_indices = indices.reshape(V3.shape[1:])
    V2_indices = np.unique(
        np.sort(
            np.concatenate(
                (V3_indices[[0, 1]], V3_indices[[0, 2]], V3_indices[[1, 2]]), axis=1
            ),
            axis=0,
        ),
        axis=1,
    )
    V2 = V1[:, V2_indices]

    Q = np.zeros((3, 3, V3.shape[2]))
    w = np.zeros((3, V3.shape[2]))

    for i in range(3):
        Q[i, i] = (
            lp.dot_VV(V3[:, (i + 1) % 3], V3[:, (i + 1) % 3])
            * lp.dot_VV(V3[:, (i + 2) % 3], V3[:, (i + 2) % 3])
            / 4
        )
        Q[i, (i + 1) % 3] = (
            lp.dot_VV(V3[:, i], V3[:, (i + 1) % 3])
            * lp.dot_VV(V3[:, (i + 2) % 3], V3[:, (i + 2) % 3])
            / 4
        )
        Q[i, (i + 2) % 3] = (
            lp.dot_VV(V3[:, i], V3[:, (i + 2) % 3])
            * lp.dot_VV(V3[:, (i + 1) % 3], V3[:, (i + 1) % 3])
            / 4
        )
        w[i] = -lp.dot_VV(V3[:, (i + 1) % 3], V3[:, (i + 2) % 3]) / 2

    omega0 = 1 / (lp.dot_VV(V2[:, 0], V2[:, 0]) * lp.dot_VV(V2[:, 1], V2[:, 1]))
    omega1 = 1 / (
        2 * np.stack([lp.dot_VV(V2[:, 0], V2[:, 0]), -lp.dot_VV(V2[:, 1], V2[:, 1])])
    )
    omega2 = 1 / (
        2 * np.stack([lp.dot_VV(V2[:, 0], V2[:, 0]), lp.dot_VV(V2[:, 1], V2[:, 1])])
    )

    return Stencil(V1, V2, V2_indices, V3, V3_indices, Q, w, omega0, omega1, omega2)


def H3(Q, w, b, delta):
    Q_delta = lp.dot_AV(Q, delta)
    r = np.sqrt(b + lp.dot_VV(delta, Q_delta))
    return np.where(np.all(Q_delta <= r * w, axis=0), r - lp.dot_VV(w, delta), -np.inf)


def H2(omega0, omega1, omega2, b, delta):
    return np.sqrt(omega0 * b + lp.dot_VV(omega1, delta) ** 2) - lp.dot_VV(
        omega2, delta
    )


def H1(v, delta):
    return -delta / lp.dot_VV(v, v)


def NewtonRoot(func, x0, params, dampingCriterion=None):
    damping = None

    if dampingCriterion is not None:
        damping = Optimization.damping_default(criterion=dampingCriterion)

    niter = 0
    spsolve_time = 0

    def Solve(residue):
        nonlocal niter
        nonlocal spsolve_time
        niter += 1
        triplets, rhs = residue.solve(raw=True)
        mat = tocsr(triplets)
        spsolve_start_time = time.process_time()
        result = spsolve(mat, rhs)
        spsolve_end_time = time.process_time()
        spsolve_time += spsolve_end_time - spsolve_start_time
        return result.reshape(residue.shape)

    newton_start_time = time.process_time()
    result = Optimization.newton_root(
        func,
        x0,
        params,
        stop=Optimization.stop_default(niter_max=100),
        damping=damping,
        solver=Solve,
    )
    newton_end_time = time.process_time()
    newton_time = newton_end_time - newton_start_time

    print(f"Newton time: {newton_time}, spsolve time: {spsolve_time}")

    if damping is None:
        return result, niter
    else:
        return result, damping.steps


def SchemeBV2(u, x, domain, A, B, u0, F, stencil):
    alpha = u.flatten()[np.argmin(lp.dot_VV(x, x))]
    u = np.where(lp.dot_VV(x, x) > np.min(lp.dot_VV(x, x)), u, u0)

    bc = Domain.Dirichlet(domain, np.inf, x)

    d2u = bc.Diff2(u, stencil.V1)
    du0 = bc.DiffUpwind(u, [[1, 0], [0, 1]])
    du1 = bc.DiffUpwind(u, [[-1, 0], [0, -1]])
    laplacian_u = np.sum(bc.Diff2(u, [[1, 0], [0, 1]]), axis=0)

    if isinstance(laplacian_u, spAD):
        laplacian_u_value = laplacian_u.value
    else:
        laplacian_u_value = laplacian_u

    du = np.where(
        np.multiply.outer(np.ones(2), laplacian_u_value) < np.inf,
        (du0 - du1) / 2,
        0,
    )

    a = np.where(
        np.multiply.outer(np.ones((2, 2)), laplacian_u_value) < np.inf,
        A(x, u, du),
        -np.inf,
    )
    spad_sum(a)

    b = np.where(laplacian_u_value < np.inf, B(x, u, du), 0)
    spad_sum(b)

    delta = d2u - lp.dot_VAV(
        np.expand_dims(stencil.V1, (2, 3)),
        np.expand_dims(a, 2),
        np.expand_dims(stencil.V1, (2, 3)),
    )
    spad_sum(delta)

    # For now, replace `b` with one when it is zero, to prevent errors during automatic
    # differentiation.
    b_zero = b == 0
    b = np.where(b_zero, 1, b)

    residue = -np.inf

    for i in range(stencil.V3.shape[2]):
        residue = np.maximum(
            residue,
            H3(
                stencil.Q[:, :, i, np.newaxis, np.newaxis],
                stencil.w[:, i, np.newaxis, np.newaxis],
                b,
                delta[stencil.V3_indices[:, i]],
            ),
        )

    for i in range(stencil.V2.shape[2]):
        residue = np.maximum(
            residue,
            H2(
                stencil.omega0[i, np.newaxis, np.newaxis],
                stencil.omega1[:, i, np.newaxis, np.newaxis],
                stencil.omega2[:, i, np.newaxis, np.newaxis],
                b,
                delta[stencil.V2_indices[:, i]],
            ),
        )

    # Reset residue to minus infinity where `b` should have been zero.
    residue = np.where(b_zero, -np.inf, residue)

    for i in range(stencil.V1.shape[1]):
        residue = np.maximum(
            residue, H1(stencil.V1[:, i, np.newaxis, np.newaxis], delta[i])
        )

    return np.where(
        bc.interior, np.maximum(residue + alpha, F(x, u, du0, du1)), u - bc.grid_values
    )


def NewtonRootBV2(u, x, domain, A, B, u0, F, stencil):
    assert np.sum(lp.dot_VV(x, x) <= np.min(lp.dot_VV(x, x))) == 1
    assert np.allclose(u.flatten()[np.argmin(lp.dot_VV(x, x))], u0)

    u, niter = NewtonRoot(
        SchemeBV2,
        np.where(lp.dot_VV(x, x) > np.min(lp.dot_VV(x, x)), u, 0),
        (x, domain, A, B, u0, F, stencil),
    )

    alpha = u.flatten()[np.argmin(lp.dot_VV(x, x))]
    u = np.where(lp.dot_VV(x, x) > np.min(lp.dot_VV(x, x)), u, u0)

    return u, alpha, niter


def SchemeDir(u, x, domain, A, B, psi, stencil):
    bc = Domain.Dirichlet(domain, psi, x)

    d2u = bc.Diff2(u, stencil.V1)

    du0, _h0 = bc.DiffUpwind(u, [[1, 0], [0, 1]], reth=True)
    du1, _h1 = bc.DiffUpwind(u, [[-1, 0], [0, -1]], reth=True)

    du = (du0 - du1) / 2

    a = A(x, u, du)
    spad_sum(a)

    b = B(x, u, du)
    spad_sum(b)

    delta = d2u - lp.dot_VAV(
        np.expand_dims(stencil.V1, (2, 3)),
        np.expand_dims(a, 2),
        np.expand_dims(stencil.V1, (2, 3)),
    )
    spad_sum(delta)

    # For now, replace `b` with one when it is zero, to prevent errors during automatic
    # differentiation.
    b_zero = b == 0
    b = np.where(b_zero, 1, b)

    residue = -np.inf

    for i in range(stencil.V3.shape[2]):
        residue = np.maximum(
            residue,
            H3(
                stencil.Q[:, :, i, np.newaxis, np.newaxis],
                stencil.w[:, i, np.newaxis, np.newaxis],
                b,
                delta[stencil.V3_indices[:, i]],
            ),
        )

    for i in range(stencil.V2.shape[2]):
        residue = np.maximum(
            residue,
            H2(
                stencil.omega0[i, np.newaxis, np.newaxis],
                stencil.omega1[:, i, np.newaxis, np.newaxis],
                stencil.omega2[:, i, np.newaxis, np.newaxis],
                b,
                delta[stencil.V2_indices[:, i]],
            ),
        )

    # Reset residue to minus infinity where `b` should have been zero.
    residue = np.where(b_zero, -np.inf, residue)

    for i in range(stencil.V1.shape[1]):
        residue = np.maximum(
            residue, H1(stencil.V1[:, i, np.newaxis, np.newaxis], delta[i])
        )

    return np.where(bc.interior, residue, u - bc.grid_values)


def NewtonRootDir(u, x, domain, A, B, psi, stencil):
    return NewtonRoot(SchemeDir, u, (x, domain, A, B, psi, stencil))


def SchemeDirLbrParts(u, x, domain, A, B, psi, stencil):
    bc = Domain.Dirichlet(domain, psi, x)

    d2u = bc.Diff2(u, stencil.V1)

    du0, _h0 = bc.DiffUpwind(u, [[1, 0], [0, 1]], reth=True)
    du1, _h1 = bc.DiffUpwind(u, [[-1, 0], [0, -1]], reth=True)

    du = (du0 - du1) / 2

    a = A(x, u, du)
    spad_sum(a)

    b = B(x, u, du)
    spad_sum(b)

    delta = d2u - lp.dot_VAV(
        np.expand_dims(stencil.V1, (2, 3)),
        np.expand_dims(a, 2),
        np.expand_dims(stencil.V1, (2, 3)),
    )
    spad_sum(delta)

    Lambda = np.inf

    for i in range(stencil.V3.shape[2]):
        m = np.maximum(0, delta[stencil.V3_indices[:, i]])
        Lambda = np.minimum(
            Lambda,
            np.where(
                m[0] >= m[1] + m[2],
                m[1] * m[2],
                np.where(
                    m[1] >= m[0] + m[2],
                    m[0] * m[2],
                    np.where(
                        m[2] >= m[0] + m[1],
                        m[0] * m[1],
                        (m[0] * m[1] + m[0] * m[2] + m[1] * m[2]) / 2
                        - (m[0] ** 2 + m[1] ** 2 + m[2] ** 2) / 4,
                    ),
                ),
            ),
        )

    return bc, b, Lambda


def SchemeDirLbr(u, x, domain, A, B, psi, stencil):
    bc, b, Lambda = SchemeDirLbrParts(u, x, domain, A, B, psi, stencil)

    return np.where(bc.interior, b - Lambda, u - bc.grid_values)


def IsInvalidDirLbr(u, x, domain, A, B, psi, stencil):
    if not isinstance(u, spAD):
        u = Sparse.identity(constant=u)

    bc, b, Lambda = SchemeDirLbrParts(u, x, domain, A, B, psi, stencil)

    return np.any(np.where(bc.interior, Lambda < b / 2, False))


def NewtonRootDirLbr(u, x, domain, A, B, psi, stencil):
    assert not IsInvalidDirLbr(u, x, domain, A, B, psi, stencil)

    return NewtonRoot(
        SchemeDirLbr,
        u,
        (x, domain, A, B, psi, stencil),
        dampingCriterion=IsInvalidDirLbr,
    )


def DiskQuad(func, start=-1):
    result, _ = dblquad(
        lambda u, v: func(np.stack([u, v])),
        start,
        1,
        lambda u: -np.sqrt(1 - u**2),
        lambda u: np.sqrt(1 - u**2),
    )

    return result


def SaveResults(filename, obj):
    with open(
        os.path.join(os.path.dirname(__file__), "../results", filename), "wb"
    ) as file:
        pickle.dump(obj, file)
