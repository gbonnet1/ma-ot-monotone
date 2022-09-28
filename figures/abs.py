#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np

import common

results = common.LoadResults("abs_sb2.pickle")

V1_sb2 = results["V1"]
x_sb2 = results["x"]
u_sb2 = results["u"]

results = common.LoadResults("abs_sb6.pickle")

V1_sb6 = results["V1"]
x_sb6 = results["x"]
u_sb6 = results["u"]

results = common.LoadResults("abs_sb10.pickle")

V1_sb10 = results["V1"]
x_sb10 = results["x"]
u_sb10 = results["u"]

x = x_sb2
assert np.allclose(x, x_sb6)
assert np.allclose(x, x_sb10)

plt.figure(figsize=(9, 6))

plt.subplot(231)
plt.plot(0, 0, ".", color="tab:blue")
plt.plot(*V1_sb2, ".", color="tab:blue")
plt.plot(*-V1_sb2, ".", color="tab:blue")
plt.title("$\\mu = 1 + \\sqrt{2}$")
plt.axis("scaled")
plt.xlim(-3.2, 3.2)
plt.ylim(-3.2, 3.2)
plt.xticks(range(-3, 4))
plt.yticks(range(-3, 4))

plt.subplot(232)
plt.plot(0, 0, ".", color="tab:blue")
plt.plot(*V1_sb6, ".", color="tab:blue")
plt.plot(*-V1_sb6, ".", color="tab:blue")
plt.title("$\\mu = 2 + \\sqrt{5}$")
plt.axis("scaled")
plt.xlim(-3.2, 3.2)
plt.ylim(-3.2, 3.2)
plt.xticks(range(-3, 4))
plt.yticks(range(-3, 4))

plt.subplot(233)
plt.plot(0, 0, ".", color="tab:blue")
plt.plot(*V1_sb10, ".", color="tab:blue")
plt.plot(*-V1_sb10, ".", color="tab:blue")
plt.title("$\\mu = 3 + \\sqrt{10}$")
plt.axis("scaled")
plt.xlim(-3.2, 3.2)
plt.ylim(-3.2, 3.2)
plt.xticks(range(-3, 4))
plt.yticks(range(-3, 4))

plt.subplot(234)
plt.contour(*x, u_sb2, levels=20)
plt.axis("scaled")
plt.xticks(np.arange(-1, 1.5, 0.5))
plt.yticks(np.arange(-1, 1.5, 0.5))

plt.subplot(235)
plt.contour(*x, u_sb6, levels=20)
plt.axis("scaled")
plt.xticks(np.arange(-1, 1.5, 0.5))
plt.yticks(np.arange(-1, 1.5, 0.5))

plt.subplot(236)
plt.contour(*x, u_sb10, levels=20)
plt.axis("scaled")
plt.xticks(np.arange(-1, 1.5, 0.5))
plt.yticks(np.arange(-1, 1.5, 0.5))

plt.subplots_adjust(
    left=0.05, bottom=0.05, right=0.98, top=0.94, wspace=0.25, hspace=0.15
)

plt.savefig(os.path.join(os.path.dirname(__file__), "out/abs.png"), dpi=300)
