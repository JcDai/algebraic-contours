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
import math

# files in the directory
from utils import *


def generate_field_collapsed_cone(workspace_path, path_to_generate_field, input_dir, input_name, output_dir, collapse_cone=True):
    print("[{}] ".format(datetime.datetime.now()),
          "Calling generate field code with collapse_cone")
    field_command = (
        path_to_generate_field
        + " --mesh "
        + input_dir
        + input_name + ".obj" + " --output " + output_dir
    )
    if collapse_cone:
        field_command += " --collapse_cones"

    print(field_command)

    subprocess.run(field_command, shell=True, check=True)


def feature_aligned_parametrization(workspace_path, path_to_feature_aligned_para, input_dir, input_name, output_dir):
    print("[{}] ".format(datetime.datetime.now()),
          "Calling feature-aligned parametrization")
    para_command = (
        path_to_feature_aligned_para
        + " --name "
        + input_name
        + " -i " + input_dir + " --use_existing_field -o " + output_dir
    )

    print(para_command)

    subprocess.run(para_command, shell=True, check=True)


def fa_para_split(workspace_path, refined_to_original_face_map_file, tet_vertices, tets, original_obj, refined_obj, surface_v_to_tet_v_map_list, surface_adj_tets, tet_surface, winding_numbers):
    surface_vertices, _, _, tris, _, _ = igl.read_obj(original_obj)
    para_vertices, _, _, para_tris, _, _ = igl.read_obj(refined_obj)

    surface_v_to_tet_v_map = {}
    for i in range(len(surface_v_to_tet_v_map_list)):
        surface_v_to_tet_v_map[i] = surface_v_to_tet_v_map_list[i]

    print("check v map validity...")
    for key in surface_v_to_tet_v_map:
        if np.linalg.norm(surface_vertices[key] - tet_vertices[surface_v_to_tet_v_map[key]]) > 1e-7:
            print("mislatch " + key + " and " + surface_v_to_tet_v_map[key])
            print(surface_vertices[key])
            print(tet_vertices[surface_v_to_tet_v_map[key]])

    print("check v map validity with tet vertices...")
    for key in surface_adj_tets:
        attached_tets = surface_adj_tets[key]

        f_vs = tris[key]
        f_vs_in_tet_base = [surface_v_to_tet_v_map[fvid] for fvid in f_vs]

        for tid in attached_tets:
            tet = tets[tid]
            for tvid in f_vs_in_tet_base:
                if tvid not in tet:
                    print("error: {} not found in tet {} ".format(tvid, tid))
                    print("f_vs: ", f_vs, " f_vs_in_tet_base: ", f_vs_in_tet_base)

    # convert refine->origin map to orignal->refine map
    refined_to_original_face_map = np.loadtxt(
        refined_to_original_face_map_file).astype(np.int32)
    para_in_to_out_face_map = {}
    for i in range(refined_to_original_face_map.shape[0]):
        if refined_to_original_face_map[i] in para_in_to_out_face_map:
            para_in_to_out_face_map[refined_to_original_face_map[i]].append(i)
        else:
            para_in_to_out_face_map[refined_to_original_face_map[i]] = [i]

    # add new vertices to tets and update map
    old_surface_v_cnt = surface_vertices.shape[0]
    new_surface_vs = para_vertices[old_surface_v_cnt:, ]

    print(np.max(para_vertices[:old_surface_v_cnt, ]-surface_vertices))
    print(np.min(para_vertices[:old_surface_v_cnt, ]-surface_vertices))

    # print(tet_vertices.shape)
    print("new_surface_vs shape: ", new_surface_vs.shape)
    # print(tet_vertices)

    for i in range(new_surface_vs.shape[0]):
        new_idx = tet_vertices.shape[0]
        tet_vertices = np.append(tet_vertices, [new_surface_vs[i]], axis=0)
        surface_v_to_tet_v_map[i + old_surface_v_cnt] = new_idx

    # print(tet_vertices)
    # print(surface_v_to_tet_v_map)

    # split tets and update maps
    new_tets = []
    new_tets_winding_numbers = []
    keep_flag = [True] * tets.shape[0]
    new_surface_adj_tets = {}

    # print(surface_adj_tets)
    print(tris.shape)

    for i in range(tris.shape[0]):
        if len(para_in_to_out_face_map[i]) == 1:
            # not splitted
            # print(i)
            new_surface_adj_tets[para_in_to_out_face_map[i]
                                 [0]] = surface_adj_tets[i]
            continue
        else:
            splitted_faces = para_in_to_out_face_map[i]
            splitted_faces_in_tet_vid = [
                [surface_v_to_tet_v_map[vid] for vid in para_tris[f]] for f in splitted_faces]

            # print(splitted_faces_in_tet_vid)

            for tet_id in surface_adj_tets[i]:
                # mark as tet to drop
                keep_flag[i] = False

                tet = tets[tet_id]
                mapped_tri = [surface_v_to_tet_v_map[k] for k in tris[i]]
                # find the apex
                apex = -1
                for vid in tet:
                    if vid not in mapped_tri:
                        apex = vid
                assert apex != -1

                for k, spf in enumerate(splitted_faces_in_tet_vid):

                    if splitted_faces[k] not in new_surface_adj_tets:
                        new_surface_adj_tets[splitted_faces[k]] = [len(
                            new_tets) + tets.shape[0]]
                    else:
                        new_surface_adj_tets[splitted_faces[k]].append(len(
                            new_tets) + tets.shape[0])
                    new_tets.append([spf[0], spf[1], spf[2], apex])
                    new_tets_winding_numbers.append(winding_numbers[tet_id])

    final_tets = []
    final_winding_numbers = []

    old_tid_to_new_tid_map = {}

    final_tet_cnt = 0
    for i in range(len(keep_flag)):
        if keep_flag[i]:
            final_tets.append(tets[i])
            final_winding_numbers.append(winding_numbers[i])
            old_tid_to_new_tid_map[i] = final_tet_cnt

        final_tet_cnt += 1

    print("kept old tets cnt: ", final_tet_cnt)

    for i in range(len(new_tets)):
        old_tid_to_new_tid_map[i + tets.shape[0]] = i + final_tet_cnt

    for key in new_surface_adj_tets:
        for i in range(len(new_surface_adj_tets[key])):
            new_surface_adj_tets[key][i] = old_tid_to_new_tid_map[new_surface_adj_tets[key][i]]

    for i in range(len(new_tets)):
        final_tets.append(np.array(new_tets[i]))
        final_winding_numbers.append(new_tets_winding_numbers[i])

    final_tets = np.array(final_tets)

    # fix orientation
    final_tets_oriented = []

    for tet in final_tets:
        if orient3d(tet_vertices[tet[0]], tet_vertices[tet[1]], tet_vertices[tet[2]], tet_vertices[tet[3]]) <= 0:
            final_tets_oriented.append(
                np.array([tet[1], tet[0], tet[2], tet[3]]))
        else:
            final_tets_oriented.append(tet)

    final_tets_oriented = np.array(final_tets_oriented)

    print("check v map validity 2...")
    for key in surface_v_to_tet_v_map:
        if np.linalg.norm(para_vertices[key] - tet_vertices[surface_v_to_tet_v_map[key]]) > 1e-7:
            print("mislatch " + str(key) + " and " +
                  str(surface_v_to_tet_v_map[key]))
            print(para_vertices[key])
            print(tet_vertices[surface_v_to_tet_v_map[key]])

    # print("check v map validity with tet vertices...")
    # for key in new_surface_adj_tets:
    #     tets = new_surface_adj_tets[key]

    #     f_vs = para_tris[key]
    #     f_vs_in_tet_base = [surface_v_to_tet_v_map[fvid] for fvid in f_vs]

    #     for tid in tets:
    #         tet = final_tets_oriented[tid]
    #         for tvid in f_vs_in_tet_base:
    #             if tvid not in tet:
    #                 print("error: {} not found in tet {} ".format(tvid, tid))
    #                 print("f_vs: ", f_vs, " f_vs_in_tet_base: ", f_vs_in_tet_base)

    return tet_vertices, final_tets_oriented, final_winding_numbers, surface_v_to_tet_v_map, new_surface_adj_tets
