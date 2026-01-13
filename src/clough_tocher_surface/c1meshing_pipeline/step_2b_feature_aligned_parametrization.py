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


def generate_field_collapsed_cone(
    workspace_path,
    path_to_generate_field,
    input_dir,
    input_name,
    output_dir,
    collapse_cone=True,
    preserve_feature=False,
):
    print(
        "[{}] ".format(datetime.datetime.now()),
        "Calling generate field code with collapse_cone",
    )
    field_command = (
        path_to_generate_field
        + " --mesh "
        + input_dir
        + input_name
        + ".obj"
        + " --output "
        + output_dir
    )
    if collapse_cone:
        field_command += " --collapse_cones"

    if not preserve_feature:
        field_command += " --feature_angle -1"

    print(field_command)

    subprocess.run(field_command, shell=True, check=True)


def feature_aligned_parametrization(
    workspace_path, path_to_feature_aligned_para, input_dir, input_name, output_dir
):
    print(
        "[{}] ".format(datetime.datetime.now()),
        "Calling feature-aligned parametrization",
    )
    # para_command = (
    #     path_to_feature_aligned_para
    #     + " --name "
    #     + input_name
    #     + " -i "
    #     + input_dir
    #     + " --use_existing_field -o "
    #     + output_dir
    # )

    para_command = (
        path_to_feature_aligned_para
        + " --name "
        + input_name
        + " -i "
        + input_dir
        + " --use_existing_field -o "
        + output_dir
        + " --show_parameterization"
    )

    print(para_command)

    subprocess.run(para_command, shell=True, check=True)


def get_feature_file(workspace_path, para_file, feature_edge_file):
    feature_edges = []
    with open(para_file, "r") as file:
        for line in file:
            if line.startswith("l "):
                tokens = line.split()
                feature_edges.append([int(tokens[1]) - 1, int(tokens[2]) - 1])

    with open(feature_edge_file, "w") as file:
        for e in feature_edges:
            file.write("{} {}\n".format(e[0], e[1]))


def fa_para_split(
    workspace_path,
    field_refined_to_original_face_map_file,
    para_refined_to_original_face_map_file,
    tet_vertices,
    tets,
    original_obj,
    refined_obj,
    surface_v_to_tet_v_map_list,
    surface_adj_tets,
    tet_surface,
    winding_numbers,
):
    surface_vertices, _, _, tris, _, _ = igl.read_obj(original_obj)
    para_vertices, _, _, para_tris, _, _ = igl.read_obj(refined_obj)

    surface_v_to_tet_v_map = {}
    for i in range(len(surface_v_to_tet_v_map_list)):
        surface_v_to_tet_v_map[i] = surface_v_to_tet_v_map_list[i]

    print("check v map validity...")
    for key in surface_v_to_tet_v_map:
        if (
            np.linalg.norm(
                surface_vertices[key] - tet_vertices[surface_v_to_tet_v_map[key]]
            )
            > 1e-7
        ):
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
    # para out to in => field_refine[para_refine]
    f_refine_map = np.loadtxt(field_refined_to_original_face_map_file).astype(np.int32)
    p_refine_map = np.loadtxt(para_refined_to_original_face_map_file).astype(np.int32)
    para_in_to_out_face_map = {}
    for i in range(p_refine_map.shape[0]):
        original_fid = f_refine_map[p_refine_map[i]]
        assert original_fid < tris.shape[0]
        if original_fid in para_in_to_out_face_map:
            para_in_to_out_face_map[original_fid].append(i)
        else:
            para_in_to_out_face_map[original_fid] = [i]

    # print(para_in_to_out_face_map[19])
    # print(para_in_to_out_face_map[61])
    # print(para_in_to_out_face_map[77])
    # print(para_in_to_out_face_map[1573])
    # print(para_in_to_out_face_map[1676])

    # add new vertices to tets and update map
    old_surface_v_cnt = surface_vertices.shape[0]
    new_surface_vs = para_vertices[old_surface_v_cnt:,]

    print(np.max(para_vertices[:old_surface_v_cnt,] - surface_vertices))
    print(np.min(para_vertices[:old_surface_v_cnt,] - surface_vertices))

    # print(tet_vertices.shape)
    print("new_surface_vs shape: ", new_surface_vs.shape)
    # print(tet_vertices)

    for i in range(new_surface_vs.shape[0]):
        new_idx = tet_vertices.shape[0]
        # print([new_surface_vs[i]])
        tet_vertices = np.append(tet_vertices, [new_surface_vs[i]], axis=0)
        # add new tet vertices to surface_v_to_tet_v_map
        surface_v_to_tet_v_map[i + old_surface_v_cnt] = new_idx

    # print(tet_vertices)
    # print(surface_v_to_tet_v_map)

    ######################################################
    ########### split tets have face on surface ##########
    ######################################################
    # split tets and update maps
    new_tets = []
    new_tets_winding_numbers = []
    keep_flag = [True for i in range(tets.shape[0])]
    new_surface_adj_tets = {}

    # print(surface_adj_tets)
    print(tris.shape)

    for i in range(tris.shape[0]):
        if len(para_in_to_out_face_map[i]) == 1:
            # not splitted
            # print(i)
            new_surface_adj_tets[para_in_to_out_face_map[i][0]] = surface_adj_tets[i]
            continue
        else:
            splitted_faces = para_in_to_out_face_map[i]
            splitted_faces_in_tet_vid = [
                [surface_v_to_tet_v_map[vid] for vid in para_tris[f]]
                for f in splitted_faces
            ]

            # print(splitted_faces_in_tet_vid)

            for tet_id in surface_adj_tets[i]:
                # mark as tet to drop
                # keep_flag[i] = False
                keep_flag[tet_id] = False

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
                        new_surface_adj_tets[splitted_faces[k]] = [
                            len(new_tets) + tets.shape[0]
                        ]
                    else:
                        new_surface_adj_tets[splitted_faces[k]].append(
                            len(new_tets) + tets.shape[0]
                        )
                    new_tets.append([spf[0], spf[1], spf[2], apex])
                    new_tets_winding_numbers.append(winding_numbers[tet_id])

    ###########################################################
    ########### split tets have only edges on surface #########
    ###########################################################

    """
    assuming each tet should have either one face or edge on the surface
    """

    # get splitted edges
    old_2_new_edge_map = {}
    for i in range(tris.shape[0]):
        if len(para_in_to_out_face_map[i]) == 1:
            # face not splitted
            continue
        else:
            # old tri vids
            old_tri_vid = tris[i]
            old_tri_vid_in_tet_vid = [
                surface_v_to_tet_v_map[vid] for vid in old_tri_vid
            ]

            # new tri vids
            splitted_faces = para_in_to_out_face_map[i]
            local_f_conn = np.array([para_tris[f] for f in splitted_faces])
            local_f_conn_in_tet_vid = np.array(
                [[surface_v_to_tet_v_map[vid] for vid in f] for f in local_f_conn]
            )

            boundary_loop = igl.boundary_loop(local_f_conn_in_tet_vid).tolist()
            assert all(old_tri_vid_in_tet_vid[lid] in boundary_loop for lid in range(3))

            if len(boundary_loop) == 3:
                # skip, no edge is splitted
                continue

            # print("original conn: ", old_tri_vid_in_tet_vid)
            # print("local conn: ")
            # print(local_f_conn_in_tet_vid)
            # print("boundary loop: ", boundary_loop)
            # print("")

            local_idx = [boundary_loop.index(vid) for vid in old_tri_vid_in_tet_vid]

            # build old to new edge map
            for k in range(3):
                v0 = old_tri_vid_in_tet_vid[k]
                v1 = old_tri_vid_in_tet_vid[(k + 1) % 3]

                v0_idx = local_idx[k]
                v1_idx = local_idx[(k + 1) % 3]

                new_edges_vs = None
                if v0_idx < v1_idx:
                    new_edges_vs = boundary_loop[v0_idx : v1_idx + 1]
                else:
                    new_edges_vs = boundary_loop[v0_idx:] + boundary_loop[: v1_idx + 1]

                if len(new_edges_vs) == 2:
                    # skip, this edge is not splitted
                    continue

                if (
                    str(v0) + "+" + str(v1) in old_2_new_edge_map
                    or str(v1) + "+" + str(v0) in old_2_new_edge_map
                ):
                    # already computed
                    continue
                else:
                    # assume same orientation, order of old tris in boundary loop will not change
                    old_2_new_edge_map[str(v0) + "+" + str(v1)] = new_edges_vs

    # find tets attach to edges
    # TODO (done): this is wrong, need to exclude the tets that are attached to faces

    # has surface flag for tets before the previous step
    tet_on_surface = [False for i in range(tets.shape[0])]
    for key in surface_adj_tets:
        for tid in surface_adj_tets[key]:
            tet_on_surface[tid] = True

    old_edge_to_tet_map = {}
    for key in old_2_new_edge_map:
        old_edge_to_tet_map[key] = []

    tet_to_splitted_edge_map = {}

    for i, tet in enumerate(tets):
        if tet_on_surface[i]:
            # skip tets have face on surface
            continue

        edges = [
            [tet[0], tet[1]],
            [tet[0], tet[2]],
            [tet[0], tet[3]],
            [tet[1], tet[2]],
            [tet[1], tet[3]],
            [tet[2], tet[3]],
        ]

        for e in edges:
            if str(e[0]) + "+" + str(e[1]) in old_edge_to_tet_map:
                old_edge_to_tet_map[str(e[0]) + "+" + str(e[1])].append(i)
                assert str(e[1]) + "+" + str(e[0]) not in old_edge_to_tet_map

                if i not in tet_to_splitted_edge_map:
                    tet_to_splitted_edge_map[i] = [str(e[0]) + "+" + str(e[1])]
                else:
                    print(
                        "after isolate surface edges each tet should have at most one surface edge"
                    )
                    assert False
                    tet_to_splitted_edge_map[i].append(str(e[0]) + "+" + str(e[1]))
            elif str(e[1]) + "+" + str(e[0]) in old_edge_to_tet_map:
                # push in to at most one direction of an edge
                old_edge_to_tet_map[str(e[1]) + "+" + str(e[0])].append(i)
                if i not in tet_to_splitted_edge_map:
                    tet_to_splitted_edge_map[i] = [str(e[1]) + "+" + str(e[0])]
                else:
                    print(
                        "after isolate surface edges each tet should have at most one surface edge"
                    )
                    assert False
                    tet_to_splitted_edge_map[i].append(str(e[1]) + "+" + str(e[0]))

    # print("edge 58-64: ", old_edge_to_tet_map["58+64"])
    # print("edge 64-58: ", old_edge_to_tet_map["64+58"])

    # split tets with edges
    new_tets_by_edges = []
    new_tets_by_edges_winding_numbers = []
    new_tets_by_edges_keep_flag = []

    for old_edge in old_edge_to_tet_map:
        old_tet_ids = old_edge_to_tet_map[old_edge]  # old tets attached to the old edge
        splitted_vids = old_2_new_edge_map[old_edge]  # v0 vnew1 vnew2 ... v1

        # print(splitted_vids)

        for old_tid in old_tet_ids:
            old_tet = tets[old_tid]

            v0_idx = old_tet.tolist().index(splitted_vids[0])
            v1_idx = old_tet.tolist().index(splitted_vids[-1])

            for k in range(len(splitted_vids) - 1):
                new_tet = [tvid for tvid in old_tet]
                new_tet[v0_idx] = splitted_vids[k]
                new_tet[v1_idx] = splitted_vids[k + 1]

                new_tet_id = len(new_tets_by_edges)

                new_tets_by_edges.append(new_tet)
                new_tets_by_edges_winding_numbers.append(winding_numbers[tet_id])
                new_tets_by_edges_keep_flag.append(True)

            keep_flag[old_tid] = False

            # delete old_tid from old_edge_to_tet_map

    # add tets splitted by edges to new tets
    new_tets += new_tets_by_edges
    new_tets_winding_numbers += new_tets_by_edges_winding_numbers

    # finalize tets

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
            new_surface_adj_tets[key][i] = old_tid_to_new_tid_map[
                new_surface_adj_tets[key][i]
            ]

    for i in range(len(new_tets)):
        final_tets.append(np.array(new_tets[i]))
        final_winding_numbers.append(new_tets_winding_numbers[i])

    final_tets = np.array(final_tets)

    # fix orientation
    final_tets_oriented = []

    for tet in final_tets:
        if (
            orient3d(
                tet_vertices[tet[0]],
                tet_vertices[tet[1]],
                tet_vertices[tet[2]],
                tet_vertices[tet[3]],
            )
            <= 0
        ):
            final_tets_oriented.append(np.array([tet[1], tet[0], tet[2], tet[3]]))
        else:
            final_tets_oriented.append(tet)

    final_tets_oriented = np.array(final_tets_oriented)

    print("check v map validity 2...")
    for key in surface_v_to_tet_v_map:
        if (
            np.linalg.norm(
                para_vertices[key] - tet_vertices[surface_v_to_tet_v_map[key]]
            )
            > 1e-7
        ):
            print("mislatch " + str(key) + " and " + str(surface_v_to_tet_v_map[key]))
            print(para_vertices[key])
            print(tet_vertices[surface_v_to_tet_v_map[key]])

    print("check v map validity with tet vertices...")
    break_flag = False
    for key in new_surface_adj_tets:
        ttt = new_surface_adj_tets[key]

        f_vs = para_tris[key]
        f_vs_in_tet_base = [surface_v_to_tet_v_map[fvid] for fvid in f_vs]

        for tid in ttt:
            tet = final_tets[tid]
            for tvid in f_vs_in_tet_base:
                if tvid not in tet:
                    print(
                        "error: {} not found in tet {}: [{}, {}, {}, {}] ".format(
                            tvid, tid, tet[0], tet[1], tet[2], tet[3]
                        )
                    )
                    print("f_vs: ", f_vs, " f_vs_in_tet_base: ", f_vs_in_tet_base)

                    break_flag = True
                    # break

            if break_flag:
                break

        if break_flag:
            break

    print("check v map validity with oriented tet vertices...")
    for key in new_surface_adj_tets:
        ttt = new_surface_adj_tets[key]

        f_vs = para_tris[key]
        f_vs_in_tet_base = [surface_v_to_tet_v_map[fvid] for fvid in f_vs]

        for tid in ttt:
            tet = final_tets_oriented[tid]
            for tvid in f_vs_in_tet_base:
                if tvid not in tet:
                    print("error: {} not found in tet {} ".format(tvid, tid))
                    print("f_vs: ", f_vs, " f_vs_in_tet_base: ", f_vs_in_tet_base)

    # test output winding >=0.5
    # print(final_tets_oriented.shape)

    tets_in_winding = []
    for i in range(final_tets_oriented.shape[0]):
        # print(final_winding_numbers[i])
        if abs(final_winding_numbers[i]) >= 0.5:
            # print(final_tets_oriented[i])
            tets_in_winding.append(final_tets_oriented[i])

    tets_in_winding = np.array(tets_in_winding)
    # print(tets_in_winding.shape)
    tetmesh_in_winding = mio.Mesh(tet_vertices, [("tetra", tets_in_winding)])
    tetmesh_in_winding.write("tet_para_split_in_winding.msh", file_format="gmsh")

    return (
        tet_vertices,
        final_tets_oriented,
        final_winding_numbers,
        surface_v_to_tet_v_map,
        new_surface_adj_tets,
    )


def trimesh_get_one_ring_vertices(vid, VF, FV):
    one_ring_vids = []
    for f in VF[vid]:
        for v in FV[f]:
            one_ring_vids.append(v)

    one_ring_vids = list(set(one_ring_vids))
    one_ring_vids.remove(vid)

    return one_ring_vids


def split_cone_one_ring(
    workspace_path,
    tets,
    tet_vertices,
    winding_numbers,
    sv_2_tv_map,
    surface_adj_tets,
    para_file,
    cone_vid_file,
):
    """
    only split the cones with 2 separate problem
    """

    # read obj
    S_V, UV_V, _, S_F, UV_F, _ = igl.read_obj(para_file)
    # read feature edges
    feature_edges = {}
    with open(para_file, "r") as file:
        for line in file:
            ss = line.split()
            if len(ss) == 3 and ss[0] == "l":
                feature_edges[str(int(ss[1]) - 1) + "+" + str(int(ss[2]) - 1)] = (
                    True  # "{e0}+{e1}"
                )

    # tetmesh connectivity initialization
    TV = tets.tolist()
    TV_preserved = [True for i in range(tets.shape[0])]
    VT = [[] for i in range(tet_vertices.shape[0])]
    V_tet = copy.deepcopy(tet_vertices)

    for tid, tet in enumerate(tets):
        for vid in tet:
            VT[vid].append(tid)

    # print(VT)

    new_winding_numbers = []  # convert winding numbers map to list
    for i in range(tets.shape[0]):
        new_winding_numbers.append(winding_numbers[i])

    # surface mesh connectivity initialization
    FV = S_F.tolist()
    FV_preserved = [True for i in range(S_F.shape[0])]
    VF = [[] for i in range(S_V.shape[0])]
    V_surface = copy.deepcopy(S_V)

    for fid, f in enumerate(S_F):
        for vid in f:
            VF[vid].append(fid)

    # uv mesh connectivity initialization
    CV = UV_F.tolist()  # c for uv
    CV_preserved = [True for i in range(UV_F.shape[0])]
    VC = [[] for i in range(UV_V.shape[0])]
    V_uv = copy.deepcopy(UV_V)

    for fid, f in enumerate(UV_F):
        for vid in f:
            VC[vid].append(fid)

    # read cone vids
    cone_vids = np.loadtxt(cone_vid_file, ndmin=1).astype(np.int32)

    # compute problematic cone vids
    adj_list = igl.adjacency_list(S_F)
    is_cone = [False for i in range(S_V.shape[0])]
    for i in cone_vids:
        is_cone[i] = True

    cone_vids_to_split = []
    for vid in range(S_V.shape[0]):
        cone_adj = []
        for adj_vid in adj_list[vid]:
            if is_cone[adj_vid]:
                cone_adj.append(adj_vid)

        if len(cone_adj) > 1:
            cone_vids_to_split.extend(cone_adj)
    cone_vids_to_split = np.unique(np.array(cone_vids_to_split))

    # print(cone_vids)

    # for each cone, split its one ring edges
    for cvid in cone_vids_to_split:
        # get one ring vertices of this cone
        surface_one_ring_vids = trimesh_get_one_ring_vertices(cvid, VF, FV)

        # iterate one ring
        for i in range(len(surface_one_ring_vids)):
            # deal with edge (v0, v1), v0 is cvid, v1 is the vid in onering
            v0 = cvid
            v1 = surface_one_ring_vids[i]

            assert v0 in sv_2_tv_map
            assert v1 in sv_2_tv_map

            # get incident faces for surface mesh
            incident_faces = list(set(VF[v0]) & set(VF[v1]))

            # print("{}-{}:\n".format(v0, v1))
            # print(VF[v0])
            # print(VF[v1])
            # print(incident_faces)
            # print("")
            assert len(incident_faces) > 0
            assert len(incident_faces) <= 2

            # prepare data for uv mesh
            # local v0 v1 ids in incident face 0 and 1
            f0_local_v0 = -1
            f0_local_v1 = -1
            f0_local_v2 = -1
            for k in range(3):
                if FV[incident_faces[0]][k] == v0:
                    f0_local_v0 = k
                elif FV[incident_faces[0]][k] == v1:
                    f0_local_v1 = k
                else:
                    f0_local_v2 = k
            assert f0_local_v0 > -1
            assert f0_local_v1 > -1
            assert f0_local_v2 > -1
            assert f0_local_v0 != f0_local_v1

            # compute f1 local vids if exist
            f1_local_v0 = -1
            f1_local_v1 = -1
            f1_local_v2 = -1

            if len(incident_faces) == 2:
                for k in range(3):
                    if FV[incident_faces[1]][k] == v0:
                        f1_local_v0 = k
                    elif FV[incident_faces[1]][k] == v1:
                        f1_local_v1 = k
                    else:
                        f1_local_v2 = k
                assert f1_local_v0 > -1
                assert f1_local_v1 > -1
                assert f1_local_v2 > -1
                assert f1_local_v0 != f1_local_v1

            #########################################
            ########### split surface mesh ##########

            # create new vertex
            new_surface_vid = V_surface.shape[0]
            new_surface_v_pos = (V_surface[v0] + V_surface[v1]) / 2.0
            V_surface = np.vstack(
                [V_surface, new_surface_v_pos]
            )  # add new surface v to V_surface
            VF.append([])  # add slot to VF

            # split incident faces
            for fid in incident_faces:
                incident_face = FV[fid]

                # get v0 v1 local id in face, also v2
                v0_local_id = -1
                v1_local_id = -1
                v2_local_id = -1
                for k in range(3):
                    if incident_face[k] == v0:
                        v0_local_id = k
                    elif incident_face[k] == v1:
                        v1_local_id = k
                    else:
                        v2_local_id = k
                assert v0_local_id > -1
                assert v1_local_id > -1
                assert v2_local_id > -1

                # remove current face
                FV_preserved[fid] = False

                # add 2 new tris
                new_fids = list(range(len(FV), len(FV) + 2))
                FV_preserved += [True, True]

                # copy original connectivity and replace the v0 v1 with v)new
                new_tri_0 = [incident_face[0], incident_face[1], incident_face[2]]
                new_tri_1 = [incident_face[0], incident_face[1], incident_face[2]]

                new_tri_0[v0_local_id] = new_surface_vid
                new_tri_1[v1_local_id] = new_surface_vid

                FV.append(new_tri_0)
                FV.append(new_tri_1)

                # add new tris to new vid VF
                VF[new_surface_vid] += new_fids

                # delete old fid from old VF
                for fvid in incident_face:
                    VF[fvid].remove(fid)

                # add new tris to old VF
                VF[incident_face[v0_local_id]].append(
                    new_fids[1]
                )  # add (v0 v_new v2) to v0
                VF[incident_face[v1_local_id]].append(
                    new_fids[0]
                )  # add (v_new v1 v2) to v1
                VF[incident_face[v2_local_id]] += [
                    new_fids[0],
                    new_fids[1],
                ]  # add two new faces to v2

            # update feature edges
            if str(v0) + "+" + str(v1) in feature_edges:
                feature_edges[str(v0) + "+" + str(v1)] = False  # delete old
                feature_edges[str(v0) + "+" + str(new_surface_vid)] = True  # add new
                feature_edges[str(new_surface_vid) + "+" + str(v1)] = True  # add new
            elif str(v1) + "+" + str(v0) in feature_edges:
                feature_edges[str(v1) + "+" + str(v0)] = False  # delete old
                feature_edges[str(v1) + "+" + str(new_surface_vid)] = True  # add new
                feature_edges[str(new_surface_vid) + "+" + str(v0)] = True  # add new

            ####################################
            ########### split uv mesh ##########

            ########### first split f_uv_0 = incident_faces[0]
            f_uv_0 = incident_faces[0]
            incident_f0 = CV[f_uv_0]
            v0_uv_f0 = incident_f0[f0_local_v0]
            v1_uv_f0 = incident_f0[f0_local_v1]

            # create and add uv f0 new vertex
            new_uv_vid_f0 = V_uv.shape[0]
            new_uv_v_pos_f0 = (V_uv[v0_uv_f0] + V_uv[v1_uv_f0]) / 2.0
            V_uv = np.vstack([V_uv, new_uv_v_pos_f0])
            VC.append([])

            # remove current uv face
            CV_preserved[f_uv_0] = False

            # add 2 new uv tris
            new_uv_fids_f0 = list(range(len(CV), len(CV) + 2))
            CV_preserved += [True, True]

            # copy original connectivity and replace the v0 v1 with v_new in uv
            new_uv_tri_f0_0 = [incident_f0[0], incident_f0[1], incident_f0[2]]
            new_uv_tri_f0_1 = [incident_f0[0], incident_f0[1], incident_f0[2]]

            new_uv_tri_f0_0[f0_local_v0] = new_uv_vid_f0
            new_uv_tri_f0_1[f0_local_v1] = new_uv_vid_f0

            CV.append(new_uv_tri_f0_0)
            CV.append(new_uv_tri_f0_1)

            # add new uv tris to new uv vid f0
            VC[new_uv_vid_f0] += new_uv_fids_f0

            # delete old uv fid from old VC
            for fuv_vid in incident_f0:
                VC[fuv_vid].remove(f_uv_0)

            # add new tris to old VC
            VC[incident_f0[f0_local_v0]].append(new_uv_fids_f0[1])  # v0 v_new v2
            VC[incident_f0[f0_local_v1]].append(new_uv_fids_f0[0])  # v_new v1 v2
            VC[incident_f0[f0_local_v2]] += [
                new_uv_fids_f0[0],
                new_uv_fids_f0[1],
            ]  # add both

            ########### split f_uv_2 if exist
            if len(incident_faces) == 2:
                f_uv_1 = incident_faces[1]
                incident_f1 = CV[f_uv_1]

                v0_uv_f1 = incident_f1[f1_local_v0]
                v1_uv_f1 = incident_f1[f1_local_v1]

                # get new vid for f1
                new_uv_vid_f1 = -1
                new_uv_v_pos_f1 = None
                if v0_uv_f1 == v0_uv_f0 and v1_uv_f1 == v1_uv_f0:
                    # if same two vertices, no need to create new vertices
                    # copy f0 new vertex
                    new_uv_vid_f1 = new_uv_vid_f0
                    new_uv_v_pos_f1 = new_uv_v_pos_f0
                else:
                    # create new uv vertex for f1
                    new_uv_vid_f1 = V_uv.shape[0]
                    new_uv_v_pos_f1 = (V_uv[v0_uv_f1] + V_uv[v1_uv_f1]) / 2.0
                    V_uv = np.vstack([V_uv, new_uv_v_pos_f1])
                    VC.append([])

                # remove current uv face
                CV_preserved[f_uv_1] = False

                # add 2 new uv tris
                new_uv_fids_f1 = list(range(len(CV), len(CV) + 2))
                CV_preserved += [True, True]

                # copy original connectivity and replace the v0 v1 with v_new in uv
                new_uv_tri_f1_0 = [incident_f1[0], incident_f1[1], incident_f1[2]]
                new_uv_tri_f1_1 = [incident_f1[0], incident_f1[1], incident_f1[2]]

                new_uv_tri_f1_0[f1_local_v0] = new_uv_vid_f1  # v_new v1 v2
                new_uv_tri_f1_1[f1_local_v1] = new_uv_vid_f1  # v0 v_new v2

                CV.append(new_uv_tri_f1_0)
                CV.append(new_uv_tri_f1_1)

                # add new uv tris to new uv vid f1
                VC[new_uv_vid_f1] += new_uv_fids_f1

                # delete old uv fid from old VC
                for fuv_vid in incident_f1:
                    VC[fuv_vid].remove(f_uv_1)

                # add new tris to old VC
                VC[incident_f1[f1_local_v0]].append(new_uv_fids_f1[1])  # v0 v_new v2
                VC[incident_f1[f1_local_v1]].append(new_uv_fids_f1[0])  # v_new v1 v2
                VC[incident_f1[f1_local_v2]] += [
                    new_uv_fids_f1[0],
                    new_uv_fids_f1[1],
                ]  # add both

            ####################################
            ########### split tet mesh #########

            # get tet v0 v1 ids
            tv0 = sv_2_tv_map[v0]
            tv1 = sv_2_tv_map[v1]

            incident_tets = list(set(VT[tv0]) & set(VT[tv1]))

            # print("tets")
            # print(VT[tv0])
            # print(VT[tv1])
            # print(incident_tets)
            assert len(incident_tets) > 0

            # create new tet vertex
            new_tet_vid = V_tet.shape[0]
            new_tet_v_pos = (V_tet[tv0] + V_tet[tv1]) / 2.0
            V_tet = np.vstack([V_tet, new_tet_v_pos])  # add to V_tet
            VT.append([])  # add slot to VT

            # update sv_2_tv_map
            # print(new_tet_v_pos)
            # print(new_surface_v_pos)
            assert np.linalg.norm(new_tet_v_pos - new_surface_v_pos) < 1e-6

            sv_2_tv_map[new_surface_vid] = new_tet_vid

            # split incident tets
            for tid in incident_tets:
                incident_tet = TV[tid]

                # print("tv0: {}, tv1: {}\n".format(tv0, tv1))
                # print("incident_tet: {}\n".format(incident_tet))

                # get tv0 tv1 local id in tet
                tv0_local_id = -1
                tv1_local_id = -1
                for k in range(4):
                    if incident_tet[k] == tv0:
                        tv0_local_id = k
                    if incident_tet[k] == tv1:
                        tv1_local_id = k
                assert tv0_local_id > -1
                assert tv1_local_id > -1

                # remove current tet
                TV_preserved[tid] = False

                # add two new tets
                new_tids = list(range(len(TV), len(TV) + 2))
                TV_preserved += [True, True]

                # copy original conn and replace tv0 and tv1
                new_tet_0 = [
                    incident_tet[0],
                    incident_tet[1],
                    incident_tet[2],
                    incident_tet[3],
                ]
                new_tet_1 = [
                    incident_tet[0],
                    incident_tet[1],
                    incident_tet[2],
                    incident_tet[3],
                ]
                new_tet_0[tv0_local_id] = new_tet_vid
                new_tet_1[tv1_local_id] = new_tet_vid

                TV.append(new_tet_0)
                TV.append(new_tet_1)

                # add 2 new winding numbers, same as old
                new_winding_numbers += [
                    new_winding_numbers[tid],
                    new_winding_numbers[tid],
                ]

                # add new tets to new vid
                VT[new_tet_vid] += new_tids

                # delete old tid from old VT
                for tvid in incident_tet:
                    VT[tvid].remove(tid)

                # add new tets to old VT
                for tvid in incident_tet:
                    if tvid != tv0 and tvid != tv1:
                        # two other vertices
                        VT[tvid] += new_tids  # add botj

                VT[tv0].append(new_tids[1])  # add v0 v_new v2 v3
                VT[tv1].append(new_tids[0])  # add v_new v1 v2 v3

    # finalize tetmesh
    final_tets = []
    final_winding_numbers = []
    for i in range(len(TV)):
        if TV_preserved[i]:
            final_tets.append(TV[i])
            final_winding_numbers.append(new_winding_numbers[i])

    final_tets = np.array(final_tets)
    final_tet_vertices = V_tet

    # finalize surface mesh
    final_tris = []
    for i in range(len(FV)):
        if FV_preserved[i]:
            final_tris.append(FV[i])

    final_tris = np.array(final_tris)
    final_tri_vertices = V_surface

    # finalize uv mesh
    final_uvs = []
    for i in range(len(CV)):
        if CV_preserved[i]:
            final_uvs.append(CV[i])

    final_uvs = np.array(final_uvs)
    final_uv_vertices = V_uv

    # build surface_adj_tet and tet_surface
    new_surface_adj_tet = {}
    new_tet_surface = {}

    tet_face_on_surface = {}
    for i, f in enumerate(final_tris):
        f_in_tvid = sorted([sv_2_tv_map[vid] for vid in f])
        f_str = str(f_in_tvid[0]) + "+" + str(f_in_tvid[1]) + "+" + str(f_in_tvid[2])
        tet_face_on_surface[f_str] = i

    for tid, tet in enumerate(final_tets):
        tet_faces = [
            sorted([tet[0], tet[1], tet[2]]),
            sorted([tet[0], tet[1], tet[3]]),
            sorted([tet[0], tet[2], tet[3]]),
            sorted([tet[1], tet[2], tet[3]]),
        ]

        for f in tet_faces:
            f_str = str(f[0]) + "+" + str(f[1]) + "+" + str(f[2])
            if f_str in tet_face_on_surface:
                fid = tet_face_on_surface[f_str]

                # update new surface_adj_tet
                if fid not in new_surface_adj_tet:
                    new_surface_adj_tet[fid] = [tid]
                else:
                    new_surface_adj_tet[fid].append(tid)
                    assert len(new_surface_adj_tet[fid]) <= 2

                # update new tet_surface
                assert tid not in new_tet_surface  # at most one face one boundary
                new_tet_surface[tid] = [fid]

    # write obj
    with open("embedded_surface_after_cone_split.obj", "w") as file:
        # write v
        for i, v in enumerate(final_tri_vertices):
            file.write("v {} {} {}\n".format(v[0], v[1], v[2]))

        # write vt
        for i, v in enumerate(final_uv_vertices):
            file.write("vt {} {}\n".format(v[0], v[1]))

        # write f
        for i, f in enumerate(final_tris):
            f_uv = final_uvs[i]
            file.write(
                "f {}/{} {}/{} {}/{}\n".format(
                    f[0] + 1, f_uv[0] + 1, f[1] + 1, f_uv[1] + 1, f[2] + 1, f_uv[2] + 1
                )
            )

    # write uv obj
    with open("embedded_surface_uv_after_cone_split.obj", "w") as file:
        # write vt
        for i, v in enumerate(final_uv_vertices):
            file.write("v {} {} 0\n".format(v[0], v[1]))

        # write f
        for i, f in enumerate(final_uvs):
            file.write("f {} {} {}\n".format(f[0] + 1, f[1] + 1, f[2] + 1))

    # write feature_edges
    with open("feature_edges_after_cone_split.txt", "w") as file:
        for key in feature_edges:
            if feature_edges[key]:
                ss = key.split("+")
                file.write("{} {}\n".format(ss[0], ss[1]))

    # write tetmesh
    tetmesh = mio.Mesh(final_tet_vertices, [("tetra", final_tets)])
    tetmesh.write("tetmesh_after_cone_split.msh", file_format="gmsh")

    return (
        final_tet_vertices,
        final_tets,
        final_winding_numbers,
        sv_2_tv_map,
        new_surface_adj_tet,
    )
