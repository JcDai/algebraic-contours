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
from step_4b_generate_CT_constraints import *
from step_5_map_nodes_tri2tet import *
from step_7b_build_hard_constraints import *
from step_6b_build_soft_constraints import *
from step_8b_polyfem import *


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

    preserve_feature = args.spec["preserve_feature"]
    cubic_optimization_step_size = args.spec["cubic_optimization_step_size"]

    path_to_generate_field_exe = args.bins["generate_field_collapsed_cones_binary"]
    path_to_feature_aligned_para_exe = args.bins[
        "feature_aligned_parametrization_binary"
    ]  # path to parametrization bin

    path_to_ct_exe = args.bins[
        "smooth_contours_binary"
    ]  # path to generate_cubic_surface bin
    path_to_ct_optimize_exe = args.bins[
        "cubic_optimization_binary"
    ]  # path to optimize_cubic_surface bin
    path_to_polyfem_exe = args.bins["polyfem_binary"]  # path to polyfem bin

    skip_cone_constraints = False

    workspace_path = ""

    start_time = time.time()
    print("start time: ", start_time)

    # step 1 read
    (
        tets,
        vertices,
        winding_numbers,
        tet_surface_origin,
        surface_adj_tet,
        para_in_v,
        para_in_f,
        para_in_v_to_tet_v_map,
        surface_tet_faces,
        surface_vertices,
    ) = read_and_generate_embedded_surface(
        workspace_path, input_file, slice=drop_unrelated_tet, debug=True
    )

    # print(para_in_v_to_tet_v_map)
    # exit(0)

    (
        tets_regular,
        tets_vertices_regular,
        surface_adj_tet,
        tet_surface,
        winding_numbers,
    ) = simplicial_embedding(
        tets,
        vertices,
        winding_numbers,
        tet_surface_origin,
        surface_adj_tet,
        surface_tet_faces,
    )

    # print(surface_adj_tet)
    # exit(0)

    shutil.copyfile("embedded_surface.obj", "embedded_surface_copy.obj")

    # step 2 feature-aligned parametrization with collapsed cones, para split
    generate_field_collapsed_cone(
        workspace_path,
        path_to_generate_field_exe,
        input_dir="./",
        input_name="embedded_surface",
        output_dir="./",
    )
    # exit(0)

    # generate frame field overwrite "embedded_surface.obj". To access original copy, use embedded_surface_copy.obj

    feature_aligned_parametrization(
        workspace_path, path_to_feature_aligned_para_exe, "./", "embedded_surface", "./"
    )

    if preserve_feature:
        get_feature_file(
            workspace_path, "embedded_surface_opt.obj", "feature_edges.txt"
        )

    # exit(0)

    (
        tet_vertices_after_para,
        tets_after_para,
        winding_numbers_after_para,
        surface_v_to_tet_v_map_after_para,
        surface_adj_tets_after_para,
    ) = fa_para_split(
        workspace_path,
        "embedded_surface_fn_to_f_field",
        "embedded_surface_fn_to_f_para",
        tets_vertices_regular,
        tets_regular,
        "embedded_surface_copy.obj",
        "embedded_surface_opt.obj",
        para_in_v_to_tet_v_map,
        surface_adj_tet,
        tet_surface_origin,
        winding_numbers,
    )
    # print(surface_adj_tets_after_para)
    # print(surface_v_to_tet_v_map_after_para)
    # exit(0)

    tetmesh_after_para = mio.Mesh(tet_vertices_after_para, [("tetra", tets_after_para)])
    tetmesh_after_para.write("test_tetmesh_after_para.msh", file_format="gmsh")
    # exit(0)

    # face split
    # print(tets_after_para[2])

    (
        tet_points_after_face_split,
        tet_cells_after_face_split,
        new_winding_numbers,
        face_split_f_to_tet_v_map,
        para_out_v_to_tet_v_map,
    ) = face_split(
        workspace_path,
        tet_vertices_after_para,
        tets_after_para,
        "embedded_surface_opt.obj",
        surface_v_to_tet_v_map_after_para,
        surface_adj_tets_after_para,
        winding_numbers_after_para,
    )

    call_gmsh(workspace_path)

    # step 4 generate CT constraints
    call_CT_code(
        workspace_path,
        path_to_ct_exe,
        "embedded_surface_opt.obj",
        skip_cons=skip_cone_constraints,
        use_initial_guess=use_initial_guess,
        preserve_feature=preserve_feature,
        feature_edge_file="feature_edges.txt",
    )

    call_CT_optimize_code(
        workspace_path,
        path_to_ct_optimize_exe,
        "embedded_surface_opt.obj",
        ct_weight,
        ct_iteration,
        use_initial_guess=use_initial_guess,
        preserve_feature=preserve_feature,
        feature_edge_file="feature_edges.txt",
        step_size=cubic_optimization_step_size,
    )

    # exit(0)

    # step 5 map tri to tet
    tet_edge_to_vertices, tet_face_to_vertices = map_tri_nodes_to_tet_nodes(
        workspace_path, output_name, face_split_f_to_tet_v_map, para_out_v_to_tet_v_map
    )

    # step 6 build soft constraints
    soft_constraint_cubic_optimization(
        workspace_path,
        output_name + "_tri_to_tet_v_map.txt",
        output_name + "_initial_tetmesh.msh",
        "CT/laplace_beltrami_mesh.msh",
        "CT/laplacian_mesh.msh",
        "CT/CT_lag2bezier_matrix.txt",
    )

    # step 7 build hard constraints

    build_full_expanded_bezier_hard_constraint_matrix(
        workspace_path,
        output_name + "_tri_to_tet_v_map.txt",
        "CT_bezier_constraints_expanded.txt",
        output_name + "_initial_tetmesh.msh",
        tet_edge_to_vertices,
        tet_face_to_vertices,
        "CT_bezier_r2f_expanded.txt",
        "CT_bezier_r2f_mat_col_idx_map.txt",
        "",
    )

    cons_time = time.time()
    print("constraints built: ", cons_time)
    print("cons took: ", cons_time - start_time)

    # step 8 polyfem
    create_polyfem_json_amips(
        enable_offset,
        output_name,
        output_name + "_initial_tetmesh.msh",
        "soft.hdf5",
        "CT_bezier_all_matrices.hdf5",
        weight_soft_1,
        elasticity_mode,
        "",
    )

    before_poly_time = time.time()
    print("before poly: ", before_poly_time)
    print("before poly took: ", before_poly_time - start_time)

    call_polyfem(workspace_path, path_to_polyfem_exe, "constraints_amips.json")

    resurrect_winding_number(output_name, new_winding_numbers, enable_offset)

    end_time = time.time()
    print("end time: ", end_time)
    print("whole took: ", end_time - start_time)
