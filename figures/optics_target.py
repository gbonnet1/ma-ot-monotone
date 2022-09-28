#!/usr/bin/env python3

import os

import imageio.v2 as imageio
import numpy as np

import common

results = common.LoadResults("optics.pickle")

target = results["target"]

m, n = target.shape
k = 10
target = np.swapaxes(
    np.multiply.outer(target, np.ones((k, k), dtype=np.uint8)), 1, 2
).reshape(m * k, n * k)
imageio.imwrite(
    os.path.join(os.path.dirname(__file__), "out/optics_target.png"), target
)
