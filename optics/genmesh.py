import io
import os
import struct

import lz4.block
import numpy as np


def genmesh(filename, object_name, vertices, faces):
    with open(
        os.path.join(os.path.dirname(__file__), "meshes", filename), "wb"
    ) as file:
        normals = []

        for i0, i1, i2 in faces:
            d1 = np.array(vertices[i1]) - np.array(vertices[i0])
            d2 = np.array(vertices[i2]) - np.array(vertices[i0])
            d3 = np.array(
                [
                    d1[1] * d2[2] - d1[2] * d2[1],
                    d1[2] * d2[0] - d1[0] * d2[2],
                    d1[0] * d2[1] - d1[1] * d2[0],
                ]
            )
            d3 = d3 / np.sqrt(np.sum(d3**2))
            normals.append((-d3[0], -d3[1], -d3[2]))

        file.write(b"BINARYMESH")
        file.write(struct.pack("<H", 4))

        b = io.BytesIO()

        b.write(struct.pack("<H", len(object_name)))
        b.write(object_name)

        b.write(struct.pack("<I", len(vertices)))
        for vertex in vertices:
            b.write(struct.pack("<fff", *vertex))

        b.write(struct.pack("<I", len(normals)))
        for normal in normals:
            b.write(struct.pack("<fff", *normal))

        b.write(struct.pack("<I", 0))

        b.write(struct.pack("<H", 1))
        b.write(struct.pack("<H", len(b"material")))
        b.write(b"material")

        b.write(struct.pack("<I", len(faces)))
        for i, (i0, i1, i2) in enumerate(faces):
            b.write(struct.pack("<H", 3))
            b.write(struct.pack("<IIi", i0, i, -1))
            b.write(struct.pack("<IIi", i1, i, -1))
            b.write(struct.pack("<IIi", i2, i, -1))

            b.write(struct.pack("<H", 0))

        buffer = b.getbuffer()
        size = len(buffer)

        compressed = lz4.block.compress(buffer)[4:]
        compressed_size = len(compressed)

        file.write(struct.pack("<QQ", size, compressed_size))
        file.write(compressed)
