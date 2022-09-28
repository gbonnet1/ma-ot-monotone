#!/usr/bin/env python3

import os
import pickle

import numpy as np
from matplotlib.tri import Triangulation

from genmesh import genmesh

with open(
    os.path.join(os.path.dirname(__file__), "../results/optics.pickle"), "rb"
) as file:
    results = pickle.load(file)

y3d = results["y3d"]
v3d = results["v3d"]

interior = np.logical_not(np.isnan(v3d))
y3d = y3d[:, interior]
v3d = v3d[interior]

tri = Triangulation(*y3d)

n_points = y3d.shape[1]
n_edges = 2 * len(tri.edges) - 3 * len(tri.triangles)

edges_dict = {}
edges = []

for i, n in enumerate(tri.triangles):
    for j in range(3):
        if tri.neighbors[i, j] == -1:
            edges_dict[n[j]] = n[(j + 1) % 3]

for i in edges_dict:
    break
j = i
while True:
    edges.append((j, edges_dict[j]))
    j = edges_dict[j]
    if j == i:
        break

vertices = []
normals = []
faces = []

for i in range(n_points):
    vertices.append((y3d[0, i], y3d[1, i], -v3d[i]))

    nrm = np.sqrt(y3d[0, i] ** 2 + y3d[1, i] ** 2 + v3d[i] ** 2)
    vertices.append(
        (
            0.9 * y3d[0, i] / nrm,
            0.9 * y3d[1, i] / nrm,
            -0.9 * v3d[i] / nrm,
        )
    )
    # vertices.append((y3d[0, i], y3d[1, i], 0.1))

for n in tri.triangles:
    faces.append([2 * n[0], 2 * n[1], 2 * n[2]])
    faces.append([2 * n[2] + 1, 2 * n[1] + 1, 2 * n[0] + 1])
for i, j in edges:
    faces.append([2 * i, 2 * i + 1, 2 * j + 1])
    faces.append([2 * i, 2 * j + 1, 2 * j])

genmesh("lens.binarymesh", b"lens", vertices, faces)
