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


def compute_cone_vids(workspace_path, cone_file="embedded_surface_Th_hat"):
    cone_angles = np.loadtxt(cone_file)

    is_cone = []
    cone_cnt = 0
    for angle in cone_angles:
        if abs(angle - 2 * math.pi) < 1e-5:
            is_cone.append(False)
        else:
            is_cone.append(True)
            cone_cnt += 1

    print("mesh has {} cones".format(cone_cnt))

    with open("para_input_cone_vids.txt", "w") as f:
        for i in range(len(is_cone)):
            if is_cone[i]:
                f.write("{}\n".format(i))


def detect_two_separate_problem(workspace_path, meshfile="embedded_surface.obj", cone_file="embedded_surface_Th_hat"):
    v, _, _, f, _, _ = igl.read_obj(meshfile)
    cone_angles = np.loadtxt(cone_file)

    is_cone = []
    cone_cnt = 0
    for angle in cone_angles:
        if abs(angle - 2 * math.pi) < 1e-5:
            is_cone.append(False)
        else:
            is_cone.append(True)
            cone_cnt += 1

    adj_list = igl.adjacency_list(f)

    problem_list = []

    for vid in range(v.shape[0]):
        cone_ids = []
        for adj_vid in adj_list[vid]:
            if is_cone[adj_vid]:
                cone_ids.append(adj_vid)
        if len(cone_ids) > 1:
            problem_list.append(vid)

    if len(problem_list) == 0:
        return True
    else:
        with open("two_separate_problem_ids.txt", "w") as file:
            for id in problem_list:
                file.write("{}\n".format(id))
        return False


def remove_adj_cones(workspace_path, meshfile="embedded_surface.obj", cone_file="embedded_surface_Th_hat"):
    v, _, _, f, _, _ = igl.read_obj(meshfile)
    cone_angles = np.loadtxt(cone_file)

    is_cone = []
    cone_cnt = 0
    cone_vids = []
    for i, angle in enumerate(cone_angles):
        if abs(angle - 2 * math.pi) < 1e-5:
            is_cone.append(False)
        else:
            is_cone.append(True)
            cone_cnt += 1
            cone_vids.append(i)

    print("initial cone cnt: ", cone_cnt)

    adj_list = igl.adjacency_list(f)

    for cid in cone_vids:
        for vid in adj_list[cid]:
            if is_cone[vid]:
                # set to 2pi
                cone_angles[cid] = 2*math.pi
                is_cone[cid] = False
                cone_cnt -= 1
                break

    for cid in cone_vids:
        if not is_cone[cid]:
            continue
        for vid in adj_list[cid]:
            assert not is_cone[vid]

    print("final cone cnt: ", cone_cnt)

    with open("embedded_surface_Th_hat_new", "w") as file:
        for angle in cone_angles:
            file.write("{}\n".format(angle))


def rearrange_cones(workspace_path, meshfile="embedded_surface.obj", cone_file="embedded_surface_Th_hat"):

    # cone 1 (cone angle alpha)  curverture = 2 * pi - alpha
    # cone 2 (cone angle beta) curverture = 2 * pi - beta
    # merge cone 1 and 2, new cone angle = 2 * pi - (2 * pi - alpha + 2 * pi - beta)

    v, _, _, f, _, _ = igl.read_obj(meshfile)
    cone_angles = np.loadtxt(cone_file)

    is_cone = []
    cone_cnt = 0
    for angle in cone_angles:
        if abs(angle - 2 * math.pi) < 1e-5:
            is_cone.append(False)
        else:
            is_cone.append(True)
            cone_cnt += 1

    adj_list = igl.adjacency_list(f)

    print("initial cone cnt: ", cone_cnt)

    satisfied = False
    while (not satisfied):
        satisfied = True
        for vid in range(v.shape[0]):
            cone_ids = []
            for adj_vid in adj_list[vid]:
                if is_cone[adj_vid]:
                    cone_ids.append(adj_vid)

            # check adj cone cnt <=1
            if len(cone_ids) > 1:
                # merge the ones sum closest to 4 * pi

                # grabbing the two rings
                for adj_vid in adj_list[vid]:
                    for two_ring_vid in adj_list[adj_vid]:
                        if is_cone[two_ring_vid]:
                            cone_ids.append(two_ring_vid)

                cone_ids = (np.unique(cone_ids)).tolist()

                print(cone_ids)
                print(cone_angles[cone_ids])

                pair2merge = (0, 1)
                for i in range(len(cone_ids)):
                    for j in range(i+1, len(cone_ids)):
                        if abs(cone_angles[cone_ids[i]] + cone_angles[cone_ids[j]] - 4 * math.pi) < abs(cone_angles[cone_ids[pair2merge[0]]] + cone_angles[cone_ids[pair2merge[1]]] - 4 * math.pi):
                            pair2merge = (i, j)

                # merge cones
                new_angle = 2 * math.pi - \
                    (2 * math.pi - cone_angles[cone_ids[pair2merge[0]]] +
                     2 * math.pi - cone_angles[cone_ids[pair2merge[1]]])

                # if (new_angle < math.pi / 2. * 2.5 or new_angle > math.pi / 2. * 8.5):
                #     satisfied = False
                #     continue

                print("merge cone {}: {} and cone {}: {} into cone {}: {}".format(
                    cone_ids[pair2merge[0]], cone_angles[cone_ids[pair2merge[0]]], cone_ids[pair2merge[1]], cone_angles[cone_ids[pair2merge[1]]], cone_ids[pair2merge[0]], new_angle))

                assert (abs(new_angle) > 1e-7)

                cone_angles[cone_ids[pair2merge[0]]] = new_angle
                cone_angles[cone_ids[pair2merge[1]]] = 2 * math.pi

                is_cone[cone_ids[pair2merge[1]]] = False

                satisfied = False

                cone_cnt -= 1
                break
            else:
                continue

    print("two-separate cone requirement satisfied.")
    print("final cone cnt: ", cone_cnt)

    with open("embedded_surface_Th_hat_new", "w") as file:
        for angle in cone_angles:
            file.write("{}\n".format(angle))

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


def parametrization_free_cones(workspace_path, path_to_para_exe, cone_file, field_file, meshfile="embedded_surface.obj"):
    print("[{}] ".format(datetime.datetime.now()),
          "Calling parametrization code")
    para_command = (
        path_to_para_exe
        + " --mesh "
        + workspace_path
        + meshfile + " --error_eps 1e-12 --use_initial_zero --use_free_cones --remove_loop_constraints" +
        " --cones " + cone_file + " --field " + field_file
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

    old_len = len(v_process) + 1

    while v_rest:
        if (old_len == len(v_rest)):
            print("cannot proceed!")
            exit()
            break

        old_len = len(v_rest)

        v_process = copy.deepcopy(v_rest)
        # v_rest = []
        v_rest = copy.deepcopy(v_process)

        # print("v_process: ", v_process)
        # print(len(v_process))

        for v in v_process:
            # assert len(adj_after[v]) == 4  # new vertices should have valence 4

            # print(v)

            # compute how many adj vertices of v is old (in before or processed)
            old_cnt = 0
            old_endpoints = []
            for v_adj in adj_after[v]:
                # if v_adj not in v_process:
                if v_adj not in v_rest:
                    old_cnt += 1
                    old_endpoints.append(v_adj)

            if old_cnt == 4:
                # there may be an old edge splitted by v
                # look for a pair of endpoints that exist in "edges" but not in "edges_after"
                found = False
                e_to_split = []
                for i in range(len(old_endpoints)):
                    vi = old_endpoints[i]
                    for j in range(i+1, len(old_endpoints)):
                        vj = old_endpoints[j]
                        eij = str(vi) + "+" + str(vj)
                        # if eij in edges and eij not in edges_after:
                        #     found = True
                        #     e_to_split = [vi, vj]

                        remaining = list(set([0, 1, 2, 3]) - set([i, j]))
                        assert len(remaining) == 2
                        vk = old_endpoints[remaining[0]]
                        vh = old_endpoints[remaining[1]]

                        if vj in adj_before[vi] and vi in adj_before[vj] and vk not in adj_before[vh] and vh not in adj_before[vk]:
                            # i and j adjcent, k and h not adjacent
                            found = True
                            e_to_split = [vi, vj]

                        if found:
                            break
                    if found:
                        break

                # print("old endpoints: ", old_endpoints)

                if found:
                    # add this edge to edges to split
                    e_to_split_with_vid = [e_to_split[0], e_to_split[1], v]
                    edges_to_split.append(e_to_split_with_vid)

                    # print("e_to_split_with_vid: ", e_to_split_with_vid)

                    # add new edges to edge dict
                    # e_adj_endpoints = list(
                    #     set(adj_before[e_to_split[0]]) & set(adj_before[e_to_split[1]]))

                    e_adj_endpoints = list(
                        set(old_endpoints) - set(e_to_split))

                    # print("adj_before e0: ", set(adj_before[e_to_split[0]]))
                    # print("adj_before e1: ", set(adj_before[e_to_split[1]]))

                    # print("e_adj_endpoints: ", e_adj_endpoints)
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

                    v_rest.remove(v)
                else:
                    # v_rest.append(v)
                    # print("cannot be processed this round inner")
                    # print(v_rest)
                    # print(v)
                    # print(adj_after[v])

                    # if v == 9564:
                    #     print(adj_before[9563])

                    assert True
            else:
                # cannot be processed this round
                # v_rest.append(v)
                # print("cannot be processed this round outer")
                assert True
                # print("")
                # print(v_rest)
                # print(v)
                # print(adj_after[v])

        # if (old_len == len(v_rest)):
        #     print("cannot proceed!")
        #     break

        # old_len = len(v_rest)

    return edges_to_split


def parametrization_split(workspace_path, tets_vertices_regular, tets_regular, surface_adj_tet, para_in_v_to_tet_v_map, path_to_toolkit_para_exe, meshfile_before_para="embedded_surface.obj", meshfile_after_para="parameterized_mesh.obj"):
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
        "output": "toolkit_para_split",
        "cone_vertices": "para_input_cone_vids.txt"
    }

    with open("para_split_json.json", "w") as f:
        json.dump(toolkit_json, f)

    print("[{}] ".format(datetime.datetime.now()),
          "Calling toolkit c1 para splitting")
    toolkit_command = (
        path_to_toolkit_para_exe + " -j " + workspace_path + "para_split_json.json"
    )
    subprocess.run(toolkit_command, shell=True, check=True)

    print("here")


def cone_split(workspace_path, path_to_toolkit_cone_exe,  para_split_tet_file, para_split_obj_file, para_split_cone_file, para_split_sf_tet_map_file, para_split_surface_adj_tet_file, para_split_sf_v_to_tet_v_file):
    tm = mio.read(para_split_tet_file)  # assuming vtu
    tm.write("cone_split_input_tetmesh.msh", file_format="gmsh")

    v, uv, _, f, fuv, _ = igl.read_obj(para_split_obj_file)
    cone_vids = np.loadtxt(para_split_cone_file).astype(np.int32)

    sm = mio.Mesh(v, [("triangle", f)])
    sm.write("cone_split_input_surface_mesh.msh", file_format="gmsh")

    uvm = mio.Mesh(uv, [("triangle", fuv)])
    uvm.write("cone_split_input_uv_mesh.msh", file_format="gmsh")

    adj_list = igl.adjacency_list(f)

    cone_edges_to_split = []
    for cid in cone_vids:
        for vid in adj_list[cid]:
            cone_edges_to_split.append([cid, vid])

    # if cone adj to cone need second split
    cone_vids_for_2nd_split = []
    for cid in cone_vids:
        for vid in adj_list[cid]:
            if vid in cone_vids:
                cone_vids_for_2nd_split.append(cid)

    with open("cone_vids_for_second_split.txt", "w") as file:
        for id in cone_vids_for_2nd_split:
            file.write("{}\n".format(id))

    # print("cone edges to split: ", cone_edges_to_split)

    with open("cone_edges_to_split.txt", "w") as file:
        for pair in cone_edges_to_split:
            file.write("{} {}\n".format(pair[0], pair[1]))

    # compute multlmesh mapping here
    surface_adj_tet_mat = np.loadtxt(
        para_split_surface_adj_tet_file).astype(np.int32)  # fid tid0 tid1
    assert surface_adj_tet_mat.shape[0] == f.shape[0]
    surface_adj_tet = [[-1, -1] for i in range(surface_adj_tet_mat.shape[0])]
    for i in range(surface_adj_tet_mat.shape[0]):
        surface_adj_tet[surface_adj_tet_mat[i][0]] = [
            surface_adj_tet_mat[i][1], surface_adj_tet_mat[i][2]]

    tets_regular = tm.cells_dict['tetra']
    para_in_v_to_tet_v_map = np.loadtxt(
        para_split_sf_v_to_tet_v_file).astype(np.int32)

    with open("toolkit_cone_split_map.txt", "w") as file:
        # {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}
        for i in range(f.shape[0]):
            tid = surface_adj_tet[i][0]
            tet = tets_regular[tid]
            f_sf_base = f[i]
            f_tet_base = np.array([para_in_v_to_tet_v_map[f_sf_base[i]]
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

    toolkit_json = {
        "tetmesh": "cone_split_input_tetmesh.msh",
        "surface_mesh": "cone_split_input_surface_mesh.msh",
        "uv_mesh": "cone_split_input_uv_mesh.msh",
        # "tet_surface_map": para_split_sf_tet_map_file,
        "tet_surface_map": "toolkit_cone_split_map.txt",
        "cone_edges": "cone_edges_to_split.txt",
        "output": "toolkit_cone_split",
        "cone_vertices": "cone_vids_for_second_split.txt"
    }

    with open("cone_split_json.json", "w") as file:
        json.dump(toolkit_json, file)

    print("[{}] ".format(datetime.datetime.now()),
          "Calling toolkit c1 cone splitting")
    toolkit_command = (
        path_to_toolkit_cone_exe + " -j " + workspace_path + "cone_split_json.json"
    )
    subprocess.run(toolkit_command, shell=True, check=True)


def cone_split_once_only(workspace_path, path_to_toolkit_cone_exe,  para_split_tet_file, para_split_obj_file, para_cone_angle_file, para_split_sf_tet_map_file, para_split_surface_adj_tet_file, para_split_sf_v_to_tet_v_file):
    tm = mio.read(para_split_tet_file)  # assuming vtu
    tm.write("cone_split_input_tetmesh.msh", file_format="gmsh")

    v, uv, _, f, fuv, _ = igl.read_obj(para_split_obj_file)

    sm = mio.Mesh(v, [("triangle", f)])
    sm.write("cone_split_input_surface_mesh.msh", file_format="gmsh")

    uvm = mio.Mesh(uv, [("triangle", fuv)])
    uvm.write("cone_split_input_uv_mesh.msh", file_format="gmsh")

    adj_list = igl.adjacency_list(f)

    cone_angles = np.loadtxt(para_cone_angle_file)
    is_cone = []
    cone_cnt = 0
    cone_vids = []
    for i, angle in enumerate(cone_angles):
        if abs(angle - 2 * math.pi) < 1e-5:
            is_cone.append(False)
        else:
            is_cone.append(True)
            cone_cnt += 1
            cone_vids.append(i)

    cone_vids_to_split = []
    for vid in range(v.shape[0]):
        cone_adj = []
        for adj_vid in adj_list[vid]:
            if is_cone[adj_vid]:
                cone_adj.append(adj_vid)

        if (len(cone_adj) > 1):
            cone_vids_to_split.extend(cone_adj)
    cone_vids_to_split = np.unique(np.array(cone_vids_to_split))

    cone_edges_to_split = []
    for cid in cone_vids_to_split:
        for vid in adj_list[cid]:
            cone_edges_to_split.append([cid, vid])

    cone_vids_for_2nd_split = []

    with open("cone_vids_for_second_split.txt", "w") as file:
        for id in cone_vids_for_2nd_split:
            file.write("{}\n".format(id))

    # print("cone edges to split: ", cone_edges_to_split)

    with open("cone_edges_to_split.txt", "w") as file:
        for pair in cone_edges_to_split:
            file.write("{} {}\n".format(pair[0], pair[1]))

    # compute multlmesh mapping here
    surface_adj_tet_mat = np.loadtxt(
        para_split_surface_adj_tet_file).astype(np.int32)  # fid tid0 tid1
    assert surface_adj_tet_mat.shape[0] == f.shape[0]
    surface_adj_tet = [[-1, -1] for i in range(surface_adj_tet_mat.shape[0])]
    for i in range(surface_adj_tet_mat.shape[0]):
        surface_adj_tet[surface_adj_tet_mat[i][0]] = [
            surface_adj_tet_mat[i][1], surface_adj_tet_mat[i][2]]

    tets_regular = tm.cells_dict['tetra']
    para_in_v_to_tet_v_map = np.loadtxt(
        para_split_sf_v_to_tet_v_file).astype(np.int32)

    with open("toolkit_cone_split_map.txt", "w") as file:
        # {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}
        for i in range(f.shape[0]):
            tid = surface_adj_tet[i][0]
            tet = tets_regular[tid]
            f_sf_base = f[i]
            f_tet_base = np.array([para_in_v_to_tet_v_map[f_sf_base[i]]
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

    toolkit_json = {
        "tetmesh": "cone_split_input_tetmesh.msh",
        "surface_mesh": "cone_split_input_surface_mesh.msh",
        "uv_mesh": "cone_split_input_uv_mesh.msh",
        # "tet_surface_map": para_split_sf_tet_map_file,
        "tet_surface_map": "toolkit_cone_split_map.txt",
        "cone_edges": "cone_edges_to_split.txt",
        "output": "toolkit_cone_split",
        "cone_vertices": "cone_vids_for_second_split.txt"
    }

    with open("cone_split_json.json", "w") as file:
        json.dump(toolkit_json, file)

    print("[{}] ".format(datetime.datetime.now()),
          "Calling toolkit c1 cone splitting")
    toolkit_command = (
        path_to_toolkit_cone_exe + " -j " + workspace_path + "cone_split_json.json"
    )
    subprocess.run(toolkit_command, shell=True, check=True)
