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
import shutil

# files in the directory
from utils import *
from step_1_generate_embedded_mesh import *
from step_2b_feature_aligned_parametrization import *
from step_3b_face_split import *
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

    ct_weight = args.spec["cubic_optimization_weight"]
    ct_iteration = args.spec["cubic_optimization_iterations"]

    initial_guess_weight = args.spec["interp_alpha"]
    use_initial_guess = args.spec["use_degenerated_initial_solution"]

    path_to_generate_field_exe = args.bins[
        "generate_field_collapsed_cones_binary"
    ]
    path_to_feature_aligned_para_exe = args.bins[
        "feature_aligned_parametrization_binary"
    ]  # path to parametrization bin

    path_to_ct_exe = args.bins[
        "smooth_contours_binary"
    ]  # path to Clough Tocher constraints bin
    path_to_polyfem_exe = args.bins["polyfem_binary"]  # path to polyfem bin

    workspace_path = ""

    start_time = time.time()
    print("start time: ", start_time)

    # step 1 read
    tets, vertices, winding_numbers, tet_surface_origin, surface_adj_tet, para_in_v, para_in_f, para_in_v_to_tet_v_map, surface_tet_faces, surface_vertices = read_and_generate_embedded_surface(
        workspace_path, input_file, slice=drop_unrelated_tet, debug=True)

    # print(para_in_v_to_tet_v_map)
    # exit(0)

    tets_regular, tets_vertices_regular, surface_adj_tet, tet_surface, winding_numbers = simplicial_embedding(
        tets, vertices, winding_numbers, tet_surface_origin, surface_adj_tet, surface_tet_faces)

    # print(surface_adj_tet)
    # exit(0)

    shutil.copyfile("embedded_surface.obj", "embedded_surface_copy.obj")

    # step 2 feature-aligned parametrization with collapsed cones, para split
    generate_field_collapsed_cone(workspace_path, path_to_generate_field_exe,
                                  input_dir="./", input_name="embedded_surface", output_dir="./")
    # exit(0)

    feature_aligned_parametrization(
        workspace_path, path_to_feature_aligned_para_exe, "./", "embedded_surface", "./")

    # exit(0)

    tet_vertices_after_para, tets_after_para, winding_numbers_after_para, surface_v_to_tet_v_map_after_para, surface_adj_tets_after_para = fa_para_split(workspace_path, "embedded_surface_fn_to_f", tets_vertices_regular, tets_regular,
                                                                                                                                                         "embedded_surface_copy.obj", "embedded_surface_opt.obj", para_in_v_to_tet_v_map, surface_adj_tet, tet_surface_origin, winding_numbers)
    # print(surface_adj_tets_after_para)
    # print(surface_v_to_tet_v_map_after_para)
    exit(0)

    tetmesh_after_para = mio.Mesh(tet_vertices_after_para, [
                                  ('tetra', tets_after_para)])
    tetmesh_after_para.write("test_tetmesh_after_para.msh", file_format="gmsh")

    # exit()
    # face split
    # print(tets_after_para[2])

    tet_points_after_face_split, tet_cells_after_face_split, new_winding_numbers, face_split_f_to_tet_v_map, para_out_v_to_tet_v_map = face_split(
        workspace_path, tet_vertices_after_para, tets_after_para, "embedded_surface_opt.obj", surface_v_to_tet_v_map_after_para, surface_adj_tets_after_para, winding_numbers_after_para)

    call_gmsh(workspace_path)

    # step 4 generate CT constraints
