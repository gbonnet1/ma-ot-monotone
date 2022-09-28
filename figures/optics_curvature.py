#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import common

results = common.LoadResults("optics.pickle")

y3d = results["y3d"]
v3d = results["v3d"]
v3dcurv = results["v3dcurv"]

plt.figure(figsize=(9, 3.5))

plt.subplot(121)
plt.axis("equal")
im = plt.pcolormesh(
    *y3d,
    v3d,
    cmap="Blues",
    shading="auto",
)
plt.colorbar(im)

plt.subplot(122)
plt.axis("equal")
im = plt.pcolormesh(
    *y3d,
    v3dcurv,
    norm=colors.BoundaryNorm(
        boundaries=np.quantile(v3dcurv[np.isfinite(v3dcurv)], np.linspace(0, 1, 256)),
        ncolors=256,
    ),
    cmap="Blues",
    shading="auto",
)
plt.colorbar(im)

plt.tight_layout()

plt.savefig(
    os.path.join(os.path.dirname(__file__), "out/optics_curvature.png"), dpi=300
)
