import igl
import meshio as mio
import numpy as np
import copy
import subprocess
import sys
import gmsh
import h5py
from scipy import sparse
import os
import json
import scipy
import datetime

# files in the directory
from utils import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-w", "--winding_number")
    parser.add_argument("-o", "--output")
    parser.add_argument("--offset", required=False, default=False)

    args = parser.parse_args()

    input = args.input
    winding = args.winding_number
    output = args.output
    enable_offset = args.offset

    polyfem_mesh = mio.read(input)

    new_winding_numbers = {}
    with open(winding, "r") as f:
        for line in f:
            kv = line.split()
            new_winding_numbers[int(kv[0])] = [float(kv[1])]

    new_winding_number_list = []
    for i in range(len(polyfem_mesh.cells[0])):
        assert i in new_winding_numbers
        new_winding_number_list.append(new_winding_numbers[i])
    new_winding_number_list = np.array(new_winding_number_list)

    new_winding_number_total = None
    if enable_offset:
        new_winding_number_total = np.zeros(
            (len(polyfem_mesh.cells[0]) + len(polyfem_mesh.cells[1]), 1)
        )
    else:
        new_winding_number_total = np.zeros(
            (len(polyfem_mesh.cells[0]), 1)
        )
    new_winding_number_total[: len(
        polyfem_mesh.cells[0])] = new_winding_number_list

    polyfem_mesh.cell_data["winding"] = new_winding_number_total[:, None]
    polyfem_mesh.write(output + "_winding.vtu")
    print("assign winding done")
