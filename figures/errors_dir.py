#!/usr/bin/env python3

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import common

results = common.LoadResults("errors_quartic_sb2_dir.pickle")

N_quartic_sb2_dir = results["N"]
errors_u_quartic_sb2_dir = results["errors_u"]
errors_du_quartic_sb2_dir = results["errors_du"]

results = common.LoadResults("errors_quartic_sb6_dir.pickle")

N_quartic_sb6_dir = results["N"]
errors_u_quartic_sb6_dir = results["errors_u"]
errors_du_quartic_sb6_dir = results["errors_du"]

results = common.LoadResults("errors_quartic_sb10_dir.pickle")

N_quartic_sb10_dir = results["N"]
errors_u_quartic_sb10_dir = results["errors_u"]
errors_du_quartic_sb10_dir = results["errors_du"]

results = common.LoadResults("errors_c1_sb2_dir.pickle")

N_c1_sb2_dir = results["N"]
errors_u_c1_sb2_dir = results["errors_u"]
errors_du_c1_sb2_dir = results["errors_du"]

results = common.LoadResults("errors_c1_sb6_dir.pickle")

N_c1_sb6_dir = results["N"]
errors_u_c1_sb6_dir = results["errors_u"]
errors_du_c1_sb6_dir = results["errors_du"]

results = common.LoadResults("errors_c1_sb10_dir.pickle")

N_c1_sb10_dir = results["N"]
errors_u_c1_sb10_dir = results["errors_u"]
errors_du_c1_sb10_dir = results["errors_du"]

results = common.LoadResults("errors_singular_sb2_dir.pickle")

N_singular_sb2_dir = results["N"]
errors_u_singular_sb2_dir = results["errors_u"]
errors_du_l1_singular_sb2_dir = results["errors_du_l1"]

results = common.LoadResults("errors_singular_sb6_dir.pickle")

N_singular_sb6_dir = results["N"]
errors_u_singular_sb6_dir = results["errors_u"]
errors_du_l1_singular_sb6_dir = results["errors_du_l1"]

results = common.LoadResults("errors_singular_sb10_dir.pickle")

N_singular_sb10_dir = results["N"]
errors_u_singular_sb10_dir = results["errors_u"]
errors_du_l1_singular_sb10_dir = results["errors_du_l1"]

N = N_c1_sb2_dir
assert np.allclose(N, N_c1_sb6_dir)
assert np.allclose(N, N_c1_sb10_dir)
assert np.allclose(N, N_singular_sb2_dir)
assert np.allclose(N, N_singular_sb6_dir)
assert np.allclose(N, N_singular_sb10_dir)

plt.figure(figsize=(9, 3.8))

plt.subplot(131)
(l0,) = plt.loglog(N, 1 / N, "w")
(l11,) = plt.loglog(N, 1 / N**2, "k--")
(l10,) = plt.loglog(N, 1 / N, "k:")
(l4,) = plt.loglog(N, errors_du_quartic_sb2_dir, ":", color="tab:orange")
(l5,) = plt.loglog(N, errors_du_quartic_sb6_dir, "--", color="tab:orange")
(l6,) = plt.loglog(N, errors_du_quartic_sb10_dir, color="tab:orange")
(l1,) = plt.loglog(N, errors_u_quartic_sb2_dir, ":", color="tab:blue")
(l2,) = plt.loglog(N, errors_u_quartic_sb6_dir, "--", color="tab:blue")
(l3,) = plt.loglog(N, errors_u_quartic_sb10_dir, color="tab:blue")
plt.title("Quartic problem")
plt.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(N)
plt.xlabel("N")
plt.ylim(3e-5, 2e-1)

plt.subplot(132)
plt.loglog(N, 1 / N**2, "k--")
plt.loglog(N, 1 / N, "k:")
plt.loglog(N, errors_du_c1_sb2_dir, ":", color="tab:orange")
plt.loglog(N, errors_du_c1_sb6_dir, "--", color="tab:orange")
plt.loglog(N, errors_du_c1_sb10_dir, color="tab:orange")
plt.loglog(N, errors_u_c1_sb2_dir, ":", color="tab:blue")
plt.loglog(N, errors_u_c1_sb6_dir, "--", color="tab:blue")
plt.loglog(N, errors_u_c1_sb10_dir, color="tab:blue")
plt.title("$C^1$ problem")
plt.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(N)
plt.xlabel("N")
plt.ylim(3e-5, 2e-1)

plt.subplot(133)
plt.loglog(N, 1 / N**2, "k--")
plt.loglog(N, 1 / N, "k:")
(l7,) = plt.loglog(N, errors_du_l1_singular_sb2_dir, ":", color="tab:red")
(l8,) = plt.loglog(N, errors_du_l1_singular_sb6_dir, "--", color="tab:red")
(l9,) = plt.loglog(N, errors_du_l1_singular_sb10_dir, color="tab:red")
plt.loglog(N, errors_u_singular_sb2_dir, ":", color="tab:blue")
plt.loglog(N, errors_u_singular_sb6_dir, "--", color="tab:blue")
plt.loglog(N, errors_u_singular_sb10_dir, color="tab:blue")
plt.title("Singular problem")
plt.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(N)
plt.xlabel("N")
plt.ylim(3e-5, 2e-1)

plt.figlegend(
    [l0, l1, l2, l3, l0, l4, l5, l6, l0, l7, l8, l9, l0, l0, l10, l11],
    [
        "$\\Vert u_h - u \\Vert_{\\infty, h}$:",
        "$\\mu = 1 + \\sqrt{2}$",
        "$\\mu = 2 + \\sqrt{5}$",
        "$\\mu = 3 + \\sqrt{10}$",
        "$\\Vert D_h u_h - D u \\Vert_{\\infty, h}$:",
        "$\\mu = 1 + \\sqrt{2}$",
        "$\\mu = 2 + \\sqrt{5}$",
        "$\\mu = 3 + \\sqrt{10}$",
        "$\\Vert D_h u_h - D u \\Vert_{1, h}$:",
        "$\\mu = 1 + \\sqrt{2}$",
        "$\\mu = 2 + \\sqrt{5}$",
        "$\\mu = 3 + \\sqrt{10}$",
        "",
        "",
        "order = 1",
        "order = 2",
    ],
    loc="lower center",
    mode="expand",
    ncol=4,
)

plt.subplots_adjust(left=0.05, bottom=0.43, right=0.99, top=0.93, wspace=0.2)

plt.savefig(os.path.join(os.path.dirname(__file__), "out/errors_dir.png"), dpi=300)
