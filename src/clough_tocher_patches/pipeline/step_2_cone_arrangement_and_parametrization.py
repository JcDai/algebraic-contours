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


def generate_frame_field(workspace_path, path_to_generate_cone_exe, meshfile="embedded_surface.obj"):
    print("[{}] ".format(datetime.datetime.now()),
          "Calling generate frame field code")
    para_command = (
        path_to_generate_cone_exe
        + " --mesh "
        + workspace_path
        + meshfile + " --input ./"
    )

    subprocess.run(para_command, shell=True, check=True)


def rearrange_cones():
    # TODO
    return


def parametrization(workspace_path, path_to_para_exe, meshfile="embedded_surface.obj", conefile="embedded_surface_Th_hat", fieldfile="embedded_surface_kappa_hat"):
    print("[{}] ".format(datetime.datetime.now()),
          "Calling parametrization code")
    para_command = (
        path_to_para_exe
        + " --mesh "
        + workspace_path
        + meshfile + " --cones " + conefile + " --field " + fieldfile
    )

    subprocess.run(
        para_command,
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.DEVNULL,
    )


def compute_edges_to_split(v_before, f_before, v_after, f_after):
    adj_before = igl.adjacency_list(f_before)
    adj_after = igl.adjacency_list(f_after)

    # new vertex ids
    v_new = list(range(v_before.shape[0], v_after.shape[0]))

    # pre-allocate
    for i in range(len(v_new)):
        adj_before.append([])

    # edges in before (and newly added edges in after in each iteration)
    edges = {}
    for v in range(len(adj_before)):
        for v_adj in adj_before[v]:
            edges[str(v) + "+" + str(v_adj)] = True

    # final edges
    edges_after = {}
    for v in range(len(adj_after)):
        for v_adj in adj_after[v]:
            edges_after[str(v) + "+" + str(v_adj)] = True

    # v_process: vids to be processed
    v_process = copy.deepcopy(v_new)

    # v_rest: vids cannot be processed in this round
    v_rest = copy.deepcopy(v_process)

    # compute edges to spilt
    edges_to_split = []

    while v_rest:
        v_process = v_rest
        v_rest = []

        for v in v_process:
            assert len(adj_after[v]) == 4  # new vertices should have valence 4

            # compute how many adj vertices of v is old (in before or processed)
            old_cnt = 0
            old_endpoints = []
            for v_adj in adj_after[v]:
                if v_adj not in v_process:
                    old_cnt += 1
                    old_endpoints.append(v_adj)

            if old_cnt >= 2:
                # there may be an old edge splitted by v
                # look for a pair of endpoints that exist in "edges" but not in "edges_after"
                found = False
                e_to_split = []
                for i in range(len(old_endpoints)):
                    vi = old_endpoints[i]
                    for j in range(i+1, len(old_endpoints)):
                        vj = old_endpoints[j]
                        eij = str(vi) + "+" + str(vj)
                        if eij in edges and eij not in edges_after:
                            found = True
                            e_to_split = [vi, vj]

                        if found:
                            break
                    if found:
                        break

                if found:
                    # add this edge to edges to split
                    e_to_split_with_vid = [e_to_split[0], e_to_split[1], v]
                    edges_to_split.append(e_to_split_with_vid)

                    # add new edges to edge dict
                    e_adj_endpoints = list(
                        set(adj_before[e_to_split[0]]) & set(adj_before[e_to_split[1]]))
                    assert len(e_adj_endpoints) == 2

                    for v_adj in e_to_split:
                        edges[str(v) + "+" + str(v_adj)] = True
                        edges[str(v_adj) + "+" + str(v)] = True

                    for v_adj in e_adj_endpoints:
                        edges[str(v) + "+" + str(v_adj)] = True
                        edges[str(v_adj) + "+" + str(v)] = True

                    # delete splitted edge from edges
                    del edges[str(e_to_split[0]) + "+" +
                              str(e_to_split[1])]
                    del edges[str(e_to_split[1]) + "+" +
                              str(e_to_split[0])]

                    # update adjacency list
                    assert len(adj_before[v]) == 0  # should be empty
                    for v_adj in e_to_split:
                        adj_before[v].append(v_adj)
                        adj_before[v_adj].append(v)

                    for v_adj in e_adj_endpoints:
                        adj_before[v].append(v_adj)
                        adj_before[v_adj].append(v)

                    # remove old edge from adjlist
                    adj_before[e_to_split[0]].remove(e_to_split[1])
                    adj_before[e_to_split[1]].remove(e_to_split[0])
                else:
                    v_rest.append(v)
            else:
                # cannot be processed this round
                v_rest.append(v)

    return edges_to_split


def parametrization_split(workspace_path, tets_vertices_regular, tets_regular, surface_adj_tet, para_in_v_to_tet_v_map, path_to_toolkit_exe, meshfile_before_para="embedded_surface.obj", meshfile_after_para="parameterized_mesh.obj"):
    # parametrization split assuming new vertices can be only added on the old edges
    v_before, _, _, f_before, _, _ = igl.read_obj(
        workspace_path + meshfile_before_para)
    v_after, uv_after, _, f_after, f_uv_after, _ = igl.read_obj(
        workspace_path + meshfile_after_para)

    if v_before.shape[0] == v_after.shape[0]:
        print("no need to do para split")

    # get multimesh map
    with open("toolkit_map.txt", "w") as file:
        # {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}
        for i in range(f_before.shape[0]):
            tid = surface_adj_tet[i][0]
            tet = tets_regular[tid]
            f = f_before[i]
            f_tet_base = np.array([para_in_v_to_tet_v_map[f[i]]
                                  for i in range(3)])
            tfs = np.array(
                [
                    [tet[1], tet[2], tet[3]],
                    [tet[0], tet[2], tet[3]],
                    [tet[0], tet[1], tet[3]],
                    [tet[0], tet[1], tet[2]],
                ]
            )
            found = False
            for k in range(4):
                if face_equal(f_tet_base, tfs[k]):
                    file.write("{} {}\n".format(tid, k))
                    found = True
            assert found

    # compute edges to split
    edges_to_split = compute_edges_to_split(
        v_before, f_before, v_after, f_after)

    print("edges to split: ", edges_to_split)

    with open(workspace_path + "toolkit_para_edges.txt", "w") as f:
        for e in edges_to_split:
            f.write("{} {} {}\n".format(e[0], e[1], e[2]))

    # prepare toolkit json
    toolkit_tet_points = tets_vertices_regular
    toolkit_tet_cells = [("tetra", tets_regular)]
    toolkit_tet = mio.Mesh(toolkit_tet_points, toolkit_tet_cells)
    toolkit_tet.write("toolkit_tet.msh", file_format="gmsh")

    toolkit_surface_points = v_before
    toolkit_surface_cells = [("triangle", f_before)]
    toolkit_surface = mio.Mesh(toolkit_surface_points, toolkit_surface_cells)
    toolkit_surface.write("toolkit_surface.msh", file_format="gmsh")

    # toolkit_uv_points = v_after
    # toolkit_uv_cells = [("triangle", p_ftc)]
    # toolkit_uv = mio.Mesh(toolkit_uv_points, toolkit_uv_cells)
    # toolkit_uv.write("toolkit_uv.msh", file_format="gmsh")

    toolkit_json = {
        "tetmesh": "toolkit_tet.msh",
        "surface_mesh": "toolkit_surface.msh",
        "uv_mesh": workspace_path + meshfile_after_para,
        "tet_surface_map": "toolkit_map.txt",
        "parametrization_edges": "toolkit_para_edges.txt",
        "output": "toolkit",
    }

    with open("para_split_json.json", "w") as f:
        json.dump(toolkit_json, f)

    print("[{}] ".format(datetime.datetime.now()),
          "Calling toolkit c1 cone splitting")
    toolkit_command = (
        path_to_toolkit_exe + " -j " + workspace_path + "para_split_json.json"
    )
    subprocess.run(toolkit_command, shell=True, check=True)

    print("here")


# # this is wrong need fix
# def parametrization_split_old(workspace_path, tets_vertices_regular, tets_regular, surface_adj_tet, para_in_v_to_tet_v_map, path_to_toolkit_exe, meshfile_before_para="embedded_surface.obj", meshfile_after_para="parameterized_mesh.obj"):
#     v_before, _, _, f_before, _, _ = igl.read_obj(
#         workspace_path + meshfile_before_para)
#     v_after, _, _, f_after, _, _ = igl.read_obj(
#         workspace_path + meshfile_after_para)

#     if v_before.shape[0] == v_after.shape[0]:
#         print("no need to do para split")
#         # return

#     # get multimesh map
#     p_v, p_tc, _, p_f, p_ftc, _ = igl.read_obj(
#         workspace_path + meshfile_after_para)
#     # TODO: this is a hack

#     with open("toolkit_map.txt", "w") as file:
#         # {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}
#         for i in range(p_f.shape[0]):
#             tid = surface_adj_tet[i][0]
#             tet = tets_regular[tid]
#             f = p_f[i]
#             f_tet_base = np.array([para_in_v_to_tet_v_map[f[i]]
#                                   for i in range(3)])
#             tfs = np.array(
#                 [
#                     [tet[1], tet[2], tet[3]],
#                     [tet[0], tet[2], tet[3]],
#                     [tet[0], tet[1], tet[3]],
#                     [tet[0], tet[1], tet[2]],
#                 ]
#             )
#             found = False
#             for k in range(4):
#                 if face_equal(f_tet_base, tfs[k]):
#                     file.write("{} {}\n".format(tid, k))
#                     found = True
#             assert found

#     # compute edges to split
#     adj_before = igl.adjacency_list(f_before)
#     adj_after = igl.adjacency_list(f_after)

#     v_new = list(range(v_before.shape[0], v_after.shape[0]))

#     v_process = copy.deepcopy(v_new)

#     v_rest = copy.deepcopy(v_process)

#     edges = {}
#     for v in range(len(adj_before)):
#         for v_adj in adj_before[v]:
#             edges[str(v) + "+" + str(v_adj)] = True

#     edges_to_split = []
#     while v_rest:
#         v_process = v_rest
#         v_rest = []
#         for v in v_process:
#             assert len(adj_after[v]) == 4
#             old_cnt = 0
#             for v_adj in adj_after[v]:
#                 if v_adj not in v_process:
#                     old_cnt += 1
#             if old_cnt > 0:
#                 # if old_cnt == 2 or old_cnt == 4:
#                 # assuming no double split on the same edge, this will give the first edges to split
#                 assert old_cnt == 2 or old_cnt == 4
#                 # find old edges
#                 old_endpoints = []
#                 for v_adj in adj_after[v]:
#                     if str(v) + "+" + str(v_adj) in edges:
#                         old_endpoints.append(v_adj)
#                 assert len(old_endpoints) == 2
#                 edges_to_split.append(old_endpoints)
#                 # add new edges to edges dict
#                 for v_adj in adj_after[v]:
#                     edges[str(v) + "+" + str(v_adj)] = True
#                     edges[str(v_adj) + "+" + str(v)] = True
#             else:
#                 v_rest.append(v)

#     print("edges to split: ", edges_to_split)
#     # assert (len(edges_to_split) > 0)

#     with open(workspace_path + "toolkit_para_edges.txt", "w") as f:
#         for e in edges_to_split:
#             f.write("{} {}\n".format(e[0], e[1]))

#     # prepare toolkit json
#     toolkit_tet_points = tets_vertices_regular
#     toolkit_tet_cells = [("tetra", tets_regular)]
#     toolkit_tet = mio.Mesh(toolkit_tet_points, toolkit_tet_cells)
#     toolkit_tet.write("toolkit_tet.msh", file_format="gmsh")

#     toolkit_surface_points = p_v
#     toolkit_surface_cells = [("triangle", p_f)]
#     toolkit_surface = mio.Mesh(toolkit_surface_points, toolkit_surface_cells)
#     toolkit_surface.write("toolkit_surface.msh", file_format="gmsh")

#     toolkit_uv_points = p_tc
#     toolkit_uv_cells = [("triangle", p_ftc)]
#     toolkit_uv = mio.Mesh(toolkit_uv_points, toolkit_uv_cells)
#     toolkit_uv.write("toolkit_uv.msh", file_format="gmsh")

#     toolkit_json = {
#         "tetmesh": "toolkit_tet.msh",
#         "surface_mesh": "toolkit_surface.msh",
#         "uv_mesh": "toolkit_uv.msh",
#         "tet_surface_map": "toolkit_map.txt",
#         "parametrization_edges": "toolkit_para_edges.txt",
#         "adjacent_cone_edges": "toolkit_cone_edges.txt",
#         "output": "toolkit",
#     }

#     with open("para_split_json.json", "w") as f:
#         json.dump(toolkit_json, f)

#     print("[{}] ".format(datetime.datetime.now()),
#           "Calling toolkit c1 cone splitting")
#     toolkit_command = (
#         path_to_toolkit_exe + " -j " + workspace_path + "para_split_json.json"
#     )
#     subprocess.run(toolkit_command, shell=True, check=True)
