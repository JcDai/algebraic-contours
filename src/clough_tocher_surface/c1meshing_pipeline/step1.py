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
import time
from argparse import ArgumentParser

# files in the directory
from utils import *
from step_1_generate_embedded_mesh import *
from step_2_cone_arrangement_and_parametrization import *
from step_3_face_split import *
from step_4_generate_CT_constraints import *
from step_5_map_nodes_tri2tet import *
from step_7_build_hard_constraints import *
from step_6_build_soft_constraints import *
from step_8_polyfem import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-j", dest="spec", required=True, type=lambda x: is_valid_json(parser, x)
    )
    parser.add_argument(
        "-b", dest="bins", required=True, type=lambda x: is_valid_json(parser, x)
    )

    args = parser.parse_args()

    input_file = args.spec[
        "input"
    ]  # vtu tetmesh file with 'winding_number' as cell data
    output_name = args.spec["output"]  # output name
    offset_file = args.spec["offset"]  # offset file
    weight_soft_1 = args.spec["weight_soft_1"]
    # int, bilaplacian on k ring 5 to 20
    k_ring_factor = args.spec["k_ring_factor"]
    sample_factor = args.spec["sample_factor"]  # int, put 2
    # LinearElasticity or Neohookean
    elasticity_mode = args.spec["elasticity_mode"]
    enable_offset = args.spec["enable_offset"]
    drop_unrelated_tet = args.spec["drop_unrelated_tet"]

    path_to_para_exe = args.bins[
        "seamless_parametrization_binary"
    ]  # path to parametrization bin
    path_to_ct_exe = args.bins[
        "smooth_contours_binary"
    ]  # path to Clough Tocher constraints bin
    path_to_polyfem_exe = args.bins["polyfem_binary"]  # path to polyfem bin
    path_to_matlab_exe = args.bins["matlab_binary"]  # path to matlab exe
    # path to toolkit app
    path_to_toolkit_exe = args.bins["wmtk_c1_cone_split_binary"]
    path_to_generate_cone_exe = args.bins["seamless_con_gen_binary"]

    workspace_path = "./"

    start_time = time.time()
    print("start time: ", start_time)

    # step 1 read
    tets, vertices, winding_numbers, tet_surface_origin, surface_adj_tet, para_in_v, para_in_f, para_in_v_to_tet_v_map, surface_tet_faces, surface_vertices = read_and_generate_embedded_surface(
        workspace_path, input_file, slice=drop_unrelated_tet, debug=True)
