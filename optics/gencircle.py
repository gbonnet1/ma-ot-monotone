#!/usr/bin/env python3

import math

from genmesh import genmesh

n = 20

vertices = [(0, 0, 0)]
for i in range(n):
    angle = -2 * i * math.pi / n
    vertices.append((math.cos(angle), math.sin(angle), 0))

faces = []
for i in range(n):
    faces.append([0, i + 1, (i + 1) % n + 1])

genmesh("circle.binarymesh", b"circle", vertices, faces)
