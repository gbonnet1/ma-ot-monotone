#!/usr/bin/env python3

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import common

results = common.LoadResults("errors_quartic_sb2.pickle")

N_quartic = results["N"]
errors_u_quartic = results["errors_u"]
errors_du_quartic = results["errors_du"]

results = common.LoadResults("errors_c1_sb2.pickle")

N_c1 = results["N"]
errors_u_c1 = results["errors_u"]
errors_du_c1 = results["errors_du"]

N = N_quartic
assert np.allclose(N, N_c1)

plt.figure(figsize=(9, 2.6))

plt.subplot(121)
(l3,) = plt.loglog(N, 1 / N, "k:")
(l2,) = plt.loglog(N, errors_du_quartic, color="tab:orange")
(l1,) = plt.loglog(N, errors_u_quartic, color="tab:blue")
plt.title("Quartic problem")
plt.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(N)
plt.xlabel("N")
plt.ylim(6e-3, 9e-1)

plt.subplot(122)
plt.loglog(N, 1 / N, "k:")
plt.loglog(N, errors_du_c1, color="tab:orange")
plt.loglog(N, errors_u_c1, color="tab:blue")
plt.title("$C^1$ problem")
plt.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(N)
plt.xlabel("N")
plt.ylim(6e-3, 9e-1)

plt.figlegend(
    [l1, l2, l3],
    [
        "$\\Vert u_h - u \\Vert_{\\infty, h}$, $\\mu = 1 + \\sqrt{2}$",
        "$\\Vert D_h u_h - D u \\Vert_{\\infty, h}$, $\\mu = 1 + \\sqrt{2}$",
        "order = 1",
    ],
    loc="center right",
)

plt.subplots_adjust(left=0.05, bottom=0.17, right=0.71, top=0.9, wspace=0.2)

plt.savefig(os.path.join(os.path.dirname(__file__), "out/errors.png"), dpi=300)
