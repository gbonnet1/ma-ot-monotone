#!/usr/bin/env python3

from genmesh import genmesh

vertices = [(0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)]
faces = [[0, 1, 2], [0, 2, 3]]

genmesh("square.binarymesh", b"square", vertices, faces)
