import numpy as np
import triangle as tr
import igl

input_file = "puzzle.obj"

v_input = []
seg = []

with open(input_file, "r") as file:
    # Read each line in the file
    for line in file:
        # print(line.strip())
        la = line.strip().split()
        if len(la) == 0:
            continue
        if la[0] == "v":
            v_input.append((float(la[1]), float(la[2])))
        if la[0] == "l":
            seg.append((int(la[1]) - 1, int(la[2]) - 1))

v_input = np.array(v_input)
seg = np.array(seg)

xmax = np.max(v_input[:, 0])
xmin = np.min(v_input[:, 0])
ymax = np.max(v_input[:, 1])
ymin = np.min(v_input[:, 1])

diag = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
pad = 0.05 * diag

v_input = np.vstack([v_input, [xmin - pad, ymin - pad]])
v_input = np.vstack([v_input, [xmin - pad, ymax + pad]])
v_input = np.vstack([v_input, [xmax + pad, ymin - pad]])
v_input = np.vstack([v_input, [xmax + pad, ymax + pad]])

# v_input = np.array(((0, 0), (1, 0), (2, 2), (0, 1)))
# seg = np.array(((0, 1), (0, 2)))

print("triangulate")
mesh = tr.triangulate(dict(vertices=v_input, segments=seg), "pcq")
print("finished")

# mesh['segments'] contains all boundaries and input segments
# mesh['segment_markers'] has 1 on all boundaries and 0 on all interior segments

# write to file
v = np.pad(mesh["vertices"], [(0, 0), (0, 1)])
f = mesh["triangles"]
igl.write_obj("t.obj", v, f)
