import meshio as mio
import numpy as np

m = mio.read("icosphere.vtu")
wn = m.cell_data["winding_number"][0].copy()

for key, _ in m.cell_data.items():
    print(key)


a = m.cell_data["winding_number"][0].copy()
a = a - np.min(a)
m.cell_data["in_out"] = a[:, None]

m.cell_data = {"in_out": m.cell_data["in_out"]}
m.write("icosphere.msh", file_format="gmsh", binary=True)
m.write("icdb.vtu")
