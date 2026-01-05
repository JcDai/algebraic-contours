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


def soft_constraint_cubic_optimization(
    workspace_path,
    tri_to_tet_index_mapping_file,
    linear_tet_file_name,
    lapbel_mesh_file,
    lap_mesh_file,
    l2b_matrix_file,
):
    local2global = np.loadtxt(workspace_path + tri_to_tet_index_mapping_file).astype(
        np.int32
    )

    m = mio.read(workspace_path + linear_tet_file_name)
    v = m.points

    # lap bel
    lapbel_mesh = mio.read(lapbel_mesh_file)
    lapbel_v = lapbel_mesh.points

    assert local2global.shape[0] == lapbel_v.shape[0]

    l2b_mat = scipy.io.mmread(l2b_matrix_file)

    A = scipy.sparse.identity(lapbel_v.shape[0]).tocoo(True)
    b = l2b_mat @ lapbel_v - v[local2global]

    with h5py.File("soft.hdf5", "w") as file:
        file.create_dataset("b", data=b)
        file.create_dataset("A_triplets/values", data=A.data)
        file.create_dataset("A_triplets/cols", data=A.col.astype(np.int32))
        file.create_dataset("A_triplets/rows", data=A.row.astype(np.int32))
        file.create_dataset("A_triplets/shape", data=A.shape)

        file.create_dataset("local2global", data=local2global.astype(np.int32))

    # lap
    lap_mesh = mio.read(lap_mesh_file)
    lap_v = lap_mesh.points

    A_2 = scipy.sparse.identity(lapbel_v.shape[0]).tocoo(True)
    b_2 = l2b_mat @ lap_v - v[local2global]

    with h5py.File("soft_lap.hdf5", "w") as file:
        file.create_dataset("b", data=b_2)
        file.create_dataset("A_triplets/values", data=A_2.data)
        file.create_dataset("A_triplets/cols", data=A_2.col.astype(np.int32))
        file.create_dataset("A_triplets/rows", data=A_2.row.astype(np.int32))
        file.create_dataset("A_triplets/shape", data=A_2.shape)

        file.create_dataset("local2global", data=local2global.astype(np.int32))


def call_CT_code_with_normals(workspace_path, path_to_ct_exe, meshfile, normals_file):
    print(
        "[{}] ".format(datetime.datetime.now()),
        "Calling Clough Tocher code with normals",
    )
    ct_command = (
        path_to_ct_exe
        + " --input "
        + workspace_path
        + meshfile
        + " -o CT "
        + "--vertex_normals "
        + normals_file
    )

    subprocess.run(ct_command, shell=True, check=True)
