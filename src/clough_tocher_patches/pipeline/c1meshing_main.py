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

    ct_weight = args.spec["cubic_optimization_weight"]
    ct_iteration = args.spec["cubic_optimization_iterations"]

    initial_guess_weight = args.spec["interp_alpha"]

    path_to_para_exe = args.bins[
        "seamless_parametrization_binary"
    ]  # path to parametrization bin
    path_to_ct_exe = args.bins[
        "smooth_contours_binary"
    ]  # path to Clough Tocher constraints bin
    path_to_polyfem_exe = args.bins["polyfem_binary"]  # path to polyfem bin
    path_to_matlab_exe = args.bins["matlab_binary"]  # path to matlab exe
    # path to toolkit app
    path_to_toolkit_para_exe = args.bins["wmtk_c1_para_split_binary"]
    path_to_toolkit_cone_exe = args.bins["wmtk_c1_cone_split_binary"]
    path_to_generate_cone_exe = args.bins["seamless_con_gen_binary"]
    path_to_ct_optimize_exe = args.bins["cubic_optimization_binary"]

    workspace_path = ""

    start_time = time.time()
    print("start time: ", start_time)

    # step 1 read
    tets, vertices, winding_numbers, tet_surface_origin, surface_adj_tet, para_in_v, para_in_f, para_in_v_to_tet_v_map, surface_tet_faces, surface_vertices = read_and_generate_embedded_surface(
        workspace_path, input_file, slice=drop_unrelated_tet, debug=True)

    tets_regular, tets_vertices_regular, surface_adj_tet, tet_surface, winding_numbers = simplicial_embedding(
        tets, vertices, winding_numbers, tet_surface_origin, surface_adj_tet, surface_tet_faces)

    # step 2 cone arrangement and parametrization
    generate_frame_field(
        workspace_path, path_to_generate_cone_exe, meshfile="embedded_surface.obj")

    remove_adj_cones(workspace_path)

    # proceed = detect_two_separate_problem(workspace_path)
    # # rearrange_cones(workspace_path)
    # # exit()
    # if not proceed:
    #     print("has vertices adjacent to two cones")
    #     # exit()

    # compute_cone_vids(workspace_path)

    # parametrization(workspace_path, path_to_para_exe, meshfile="embedded_surface.obj",
    #                 conefile="embedded_surface_Th_hat", fieldfile="embedded_surface_kappa_hat")
    # parametrization(workspace_path, path_to_para_exe, meshfile="embedded_surface.obj",
    #                 conefile="embedded_surface_Th_hat_new", fieldfile="embedded_surface_kappa_hat")

    parametrization_free_cones(
        workspace_path, path_to_para_exe, cone_file="embedded_surface_Th_hat_new", field_file="embedded_surface_kappa_hat", meshfile="embedded_surface.obj")

    # compute_cone_vids(workspace_path)
    compute_cone_vids(workspace_path, cone_file="embedded_surface_Th_hat_new")

    parametrization_split(workspace_path, tets_vertices_regular, tets_regular, surface_adj_tet, para_in_v_to_tet_v_map,
                          path_to_toolkit_para_exe, meshfile_before_para="embedded_surface.obj", meshfile_after_para="parameterized_mesh.obj")

    # cone_split(workspace_path, path_to_toolkit_cone_exe, "toolkit_para_split_tetmesh_tets.vtu",
    #            "surface_uv_after_para_split.obj", "cone_vids_after_para_split.txt", "surface_tet_local_face_map_after_para_split.txt", "surface_adj_tet_after_para_split.txt", "surface_v_to_tet_v_after_para_split.txt")

    cone_split_once_only(workspace_path, path_to_toolkit_cone_exe, "toolkit_para_split_tetmesh_tets.vtu",
                         "surface_uv_after_para_split.obj", "cone_vids_after_para_split.txt", "surface_tet_local_face_map_after_para_split.txt", "surface_adj_tet_after_para_split.txt", "surface_v_to_tet_v_after_para_split.txt")

    # exit()

    # step 3 face split
    tet_points_after_face_split, tet_cells_after_face_split, new_winding_numbers, face_split_f_to_tet_v_map, para_out_v_to_tet_v_map = face_split(workspace_path, "toolkit_cone_split_tetmesh_tets.vtu", "surface_uv_after_cone_split.obj",
                                                                                                                                                  "surface_v_to_tet_v_after_cone_split.txt", "surface_adj_tet_after_cone_split.txt")

    call_gmsh(workspace_path)

    # step 4 generate CT constraints
    call_CT_code(workspace_path, path_to_ct_exe,
                 "surface_uv_after_cone_split.obj")

    call_CT_optimize_code(workspace_path, path_to_ct_optimize_exe,
                          "surface_uv_after_cone_split.obj", ct_weight, ct_iteration)

    # step 5 map tri to tet
    tet_edge_to_vertices, tet_face_to_vertices = map_tri_nodes_to_tet_nodes(
        workspace_path, output_name, face_split_f_to_tet_v_map, para_out_v_to_tet_v_map)

    interpolate_initial_solution(output_name, "laplace_beltrami_mesh.msh", "CT_degenerate_cubic_bezier_points.msh",
                                 output_name+"_initial_tetmesh.msh", initial_guess_weight, output_name + "_tri_to_tet_v_map.txt", "CT_lag2bezier_matrix.txt")

    # interpolate_initial_solution_test(output_name, "CT_degenerate_cubic_bezier_points.msh",
    #                                   output_name+"_initial_tetmesh.msh", initial_guess_weight, output_name + "_tri_to_tet_v_map.txt", "CT_lag2bezier_matrix.txt")

    # step 6 build soft constraints
    # A_sti, b_sti = upsample_and_smooth_cones("CT_bilaplacian_nodes_values_cone_area_vertices.txt",
    #                                          "CT_bilaplacian_nodes_values_cone_area_faces.txt", "CT_from_lagrange_nodes.msh", sample_factor, k_ring_factor)

    # soft_constraint_fit_normal(workspace_path, output_name + "_tri_to_tet_v_map.txt",
    #                            output_name + "_initial_tetmesh.msh", A_sti, b_sti)

    # call_CT_code_with_normals(workspace_path, path_to_ct_exe,
    #                           "surface_uv_after_cone_split.obj", "CT_smoothed_normals.txt")

    soft_constraint_cubic_optimization(workspace_path, output_name + "_tri_to_tet_v_map.txt",
                                       output_name + "_initial_tetmesh.msh", "laplace_beltrami_mesh.msh", "laplacian_mesh.msh", "CT_lag2bezier_matrix.txt")

    # soft_constraint_cubic_optimization(workspace_path, output_name + "_tri_to_tet_v_map.txt",
    #                                    output_name + "_interp_initial_mesh.msh", "laplace_beltrami_mesh.msh", "laplacian_mesh.msh", "CT_lag2bezier_matrix.txt")

    # step 7 build hard constraints

    build_full_expanded_bezier_hard_constraint_matrix(workspace_path, output_name + "_tri_to_tet_v_map.txt",
                                                      "CT_bezier_constraints_expanded_old.txt", output_name + "_initial_tetmesh.msh", tet_edge_to_vertices, tet_face_to_vertices, "CT_bezier_r2f_expanded_old.txt", "CT_bezier_r2f_mat_col_idx_map_old.txt", output_name + "_interp_initial_mesh.msh")

    # build_full_expanded_bezier_hard_constraint_matrix(workspace_path, output_name + "_tri_to_tet_v_map.txt",
    #                                                   "CT_bezier_constraints_expanded.txt", output_name + "_initial_tetmesh.msh", tet_edge_to_vertices, tet_face_to_vertices, "CT_bezier_r2f_expanded.txt", "CT_bezier_r2f_mat_col_idx_map.txt", output_name + "_interp_initial_mesh.msh")
    # build_full_expanded_bezier_hard_constraint_matrix(workspace_path, output_name + "_tri_to_tet_v_map.txt",
    #                                                   "CT_bezier_constraints_expanded.txt", output_name + "_interp_initial_mesh.msh", tet_edge_to_vertices, tet_face_to_vertices, "CT_bezier_r2f_expanded.txt", "CT_bezier_r2f_mat_col_idx_map.txt", output_name + "_interp_initial_mesh.msh")

    cons_time = time.time()
    print("constraints built: ", cons_time)
    print("cons took: ", cons_time - start_time)

    # step 8 polyfem
    create_polyfem_json(enable_offset, output_name, output_name + "_initial_tetmesh.msh", "soft.hdf5",
                        "CT_bezier_all_matrices.hdf5", weight_soft_1, elasticity_mode, "")

    create_polyfem_json_amips(enable_offset, output_name, output_name + "_initial_tetmesh.msh", "soft.hdf5",
                              "CT_bezier_all_matrices.hdf5", weight_soft_1, elasticity_mode, "")

    # create_polyfem_json(enable_offset, output_name, output_name + "_interp_initial_mesh.msh", "soft.hdf5",
    #                     "CT_bezier_all_matrices.hdf5", weight_soft_1, elasticity_mode, "")

    # create_polyfem_json_amips(enable_offset, output_name, output_name + "_interp_initial_mesh.msh", "soft.hdf5",
    #                           "CT_bezier_all_matrices.hdf5", weight_soft_1, elasticity_mode, "")

    before_poly_time = time.time()
    print("before poly: ", before_poly_time)
    print("before poly took: ", before_poly_time - start_time)

    call_polyfem(workspace_path, path_to_polyfem_exe, "constraints_amips.json")
    # call_polyfem(workspace_path, path_to_polyfem_exe, "constraints.json")

    resurrect_winding_number(output_name, new_winding_numbers, enable_offset)

    end_time = time.time()
    print("end time: ", end_time)
    print("whole took: ", end_time - start_time)
