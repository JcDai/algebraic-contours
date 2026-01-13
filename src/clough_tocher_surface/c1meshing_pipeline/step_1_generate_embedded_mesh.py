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

# read and slice a mesh if needed, returning the v, t and winding numbers
# slice is not done now


def read_and_generate_embedded_surface(workspace_path, input, slice=False, debug=False):
    tm = mio.read(input)
    vertices_unsliced = tm.points
    tets_unsliced = tm.cells_dict["tetra"]

    if debug:
        print("[{}] ".format(datetime.datetime.now()), "checking tet orientation ...")
    for i in range(tets_unsliced.shape[0]):
        if not orient3d(
            vertices_unsliced[tets_unsliced[i][0]],
            vertices_unsliced[tets_unsliced[i][1]],
            vertices_unsliced[tets_unsliced[i][2]],
            vertices_unsliced[tets_unsliced[i][3]],
        ):
            print("tet {} flipped".format(i))
        assert orient3d(
            vertices_unsliced[tets_unsliced[i][0]],
            vertices_unsliced[tets_unsliced[i][1]],
            vertices_unsliced[tets_unsliced[i][2]],
            vertices_unsliced[tets_unsliced[i][3]],
        )
    print("[{}] ".format(datetime.datetime.now()), "passed orientation check.")

    winding_numbers_data_unsliced = tm.cell_data["winding_number"][0]

    # remove tet not touching surface
    filtered_tets = []
    for i in range(tets_unsliced.shape[0]):
        if abs(winding_numbers_data_unsliced[i]) >= 0.5:
            filtered_tets.append(tets_unsliced[i])
    filtered_tets = np.array(filtered_tets)

    surface_tet_faces = igl.boundary_facets(filtered_tets)
    surface_vertices = np.unique(surface_tet_faces)

    if not slice:
        tets = tets_unsliced
        vertices = vertices_unsliced
        winding_numbers = {}
        winding_numbers_data = winding_numbers_data_unsliced
        for i in range(tets.shape[0]):
            winding_numbers[i] = winding_numbers_data[i]
    else:
        # one ring
        tet_indices_touching_surface = np.unique(
            np.argwhere(np.isin(tets_unsliced, surface_vertices))[:, 0]
        )

        # # two ring
        # tets_one_ring = tets_unsliced[tet_indices_touching_surface]
        # vertices_one_ring = np.unique(tets_one_ring.flatten())
        # tet_indices_touching_surface = np.unique(np.argwhere(
        #     np.isin(tets_unsliced, vertices_one_ring))[:, 0])

        tets = tets_unsliced[tet_indices_touching_surface]
        winding_numbers_data = winding_numbers_data_unsliced[
            tet_indices_touching_surface
        ]
        vertices = vertices_unsliced
        vertices, tets, _, sliced_to_unsliced_v_map = igl.remove_unreferenced(
            vertices_unsliced, tets
        )

        # save the remaining part
        remaining_indices = list(
            set(range(tets_unsliced.shape[0])) - set(tet_indices_touching_surface)
        )

        tets_unused_inside = []
        winding_numbers_unused_inside = []
        tets_unused_outside = []
        winding_numbers_unused_outside = []
        for tid in remaining_indices:
            if abs(winding_numbers_data_unsliced[tid]) >= 0.5:
                tets_unused_inside.append(tets_unsliced[tid])
                winding_numbers_unused_inside.append(winding_numbers_data_unsliced[tid])
            else:
                tets_unused_outside.append(tets_unsliced[tid])
                winding_numbers_unused_outside.append(
                    winding_numbers_data_unsliced[tid]
                )

        if len(tets_unused_inside) != 0:
            unused_inside = mio.Mesh(
                vertices_unsliced, [("tetra", np.array(tets_unused_inside))]
            )
            unused_inside.write("tets_unused_inside.msh", file_format="gmsh")
        if len(tets_unused_outside) != 0:
            unused_outside = mio.Mesh(
                vertices_unsliced, [("tetra", np.array(tets_unused_outside))]
            )
            unused_outside.write("tets_unused_outside.msh", file_format="gmsh")

        # print(winding_numbers_data.shape)
        # print(winding_numbers_data)
        # print(tets.shape)

        # fix  winding number shape
        if len(winding_numbers_data.shape) == 1:
            winding_numbers_data = winding_numbers_data[:, None]
        # print(winding_numbers_data.shape)
        # print(winding_numbers_data)

        # exit()

        m_sliced = mio.Mesh(
            vertices,
            [("tetra", tets)],
            cell_data={"winding_number": winding_numbers_data.T},
        )
        m_sliced.write("test_slice.vtu")

        # extract surface
        winding_numbers = {}
        for i in range(tets.shape[0]):
            winding_numbers[i] = winding_numbers_data[i]

        surface_tet_faces_unsliced = igl.boundary_facets(filtered_tets)
        unsliced_to_sliced_v_map = {}
        for i in range(len(sliced_to_unsliced_v_map)):
            unsliced_to_sliced_v_map[sliced_to_unsliced_v_map[i]] = i
        surface_tet_faces = surface_tet_faces_unsliced.copy().tolist()
        for i in range(len(surface_tet_faces)):
            for j in range(3):
                # print(surface_tet_face[i][j])
                surface_tet_faces[i][j] = unsliced_to_sliced_v_map[
                    surface_tet_faces[i][j]
                ]
        surface_tet_faces = np.array(surface_tet_faces)
        print("!!!!!!!", surface_tet_faces.shape)

    para_in_v, para_in_f, im, para_in_v_to_tet_v_map = igl.remove_unreferenced(
        vertices, surface_tet_faces
    )
    print("??????", para_in_f.shape)

    assert (igl.bfs_orient(para_in_f)[0] == para_in_f).all()

    igl.write_obj(workspace_path + "embedded_surface.obj", para_in_v, para_in_f)
    print(
        "[{}] ".format(datetime.datetime.now()),
        "generated embedded_surface.obj for parametrization.",
    )

    # get tets containing surface
    print(
        "[{}] ".format(datetime.datetime.now()),
        "constructing tet->surface and surface->tet mapping ...",
    )
    surface_in_tet_map = {}
    for i in range(tets.shape[0]):
        ff0 = [tets[i][0], tets[i][1], tets[i][2]]
        ff1 = [tets[i][0], tets[i][1], tets[i][3]]
        ff2 = [tets[i][0], tets[i][2], tets[i][3]]
        ff3 = [tets[i][1], tets[i][2], tets[i][3]]
        ff0.sort()
        ff1.sort()
        ff2.sort()
        ff3.sort()

        ffs = [
            str(ff0[0]) + "+" + str(ff0[1]) + "+" + str(ff0[2]),
            str(ff1[0]) + "+" + str(ff1[1]) + "+" + str(ff1[2]),
            str(ff2[0]) + "+" + str(ff2[1]) + "+" + str(ff2[2]),
            str(ff3[0]) + "+" + str(ff3[1]) + "+" + str(ff3[2]),
        ]
        for f_str in ffs:
            if f_str in surface_in_tet_map:
                surface_in_tet_map[f_str].append(i)
            else:
                surface_in_tet_map[f_str] = [i]

    print("[{}] ".format(datetime.datetime.now()), "computed face in tet map")

    surface_adj_tet = {}
    tet_surface_origin = {}

    for i in range(surface_tet_faces.shape[0]):
        surface_adj_tet[i] = []

    for j in range(tets.shape[0]):
        tet_surface_origin[j] = []

    for i in range(surface_tet_faces.shape[0]):
        face = [
            surface_tet_faces[i][0],
            surface_tet_faces[i][1],
            surface_tet_faces[i][2],
        ]
        face.sort()
        face_str = str(face[0]) + "+" + str(face[1]) + "+" + str(face[2])
        surface_adj_tet[i] = surface_in_tet_map[face_str]
        assert len(surface_adj_tet[i]) > 0
        for tt in surface_in_tet_map[face_str]:
            tet_surface_origin[tt].append(i)

    print(
        "[{}] ".format(datetime.datetime.now()),
        "computed tet->surface and surface->tet mapping.",
    )

    # print(surface_adj_tet)

    return (
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
    )


# simplicial embedding
def simplicial_embedding(
    tets,
    vertices,
    winding_numbers,
    tet_surface_origin,
    surface_adj_tet,
    surface_tet_faces,
):
    print("[{}] ".format(datetime.datetime.now()), "Doing simplicial embedding ...")
    tets_regular = copy.deepcopy(tets).tolist()
    tets_vertices_regular = copy.deepcopy(vertices).tolist()
    tet_surface = copy.deepcopy(tet_surface_origin)

    tet_surface = copy.deepcopy(tet_surface_origin)

    simplicial_embedding_cnt = 0
    for i in range(tets.shape[0]):
        if len(tet_surface[i]) > 1:
            simplicial_embedding_cnt += 1
            # tet contain more than one surface, need split into 4
            fs = [f for f in tet_surface[i]]

            # create new vertex
            new_v_id = len(tets_vertices_regular)
            v_new = (
                vertices[tets[i][0]]
                + vertices[tets[i][1]]
                + vertices[tets[i][2]]
                + vertices[tets[i][3]]
            ) / 4.0
            tets_vertices_regular.append(v_new.tolist())

            # create new tets
            new_t_ids = [
                i,
                len(tets_regular),
                len(tets_regular) + 1,
                len(tets_regular) + 2,
            ]
            tets_regular[i] = [new_v_id, tets[i][1], tets[i][2], tets[i][3]]
            tets_regular.append([tets[i][0], new_v_id, tets[i][2], tets[i][3]])
            tets_regular.append([tets[i][0], tets[i][1], new_v_id, tets[i][3]])
            tets_regular.append([tets[i][0], tets[i][1], tets[i][2], new_v_id])

            # propagate winding number
            old_winding_number = winding_numbers[i]
            for tid in new_t_ids:
                winding_numbers[tid] = old_winding_number

            # reset tet surface mapping
            for tid in new_t_ids:
                tet_surface[tid] = []

            # update map
            for f in fs:
                surface_adj_tet[f].remove(i)
                for tid in new_t_ids:
                    if face_in_tet(surface_tet_faces[f], tets_regular[tid]):
                        surface_adj_tet[f].append(tid)
                        tet_surface[tid] = [f]
                        break

    tets_regular = np.array(tets_regular)
    tets_vertices_regular = np.array(tets_vertices_regular)

    print(
        "[{}] ".format(datetime.datetime.now()),
        "Done simplicial embedding. Splitted {} tets".format(simplicial_embedding_cnt),
    )

    return (
        tets_regular,
        tets_vertices_regular,
        surface_adj_tet,
        tet_surface,
        winding_numbers,
    )


# simplicial embedding isolating surface edges


def isolate_surface_edges(
    tets, vertices, surface_adj_tet, tet_surface, surface_tet_faces, winding_numbers
):
    """
    split tets that has more than one edge on the surface, assuming input tet has at most one face on the surface

    :param tets: TV conn
    :param vertices: V
    :param surface_adj_tet: surface fid to attaching tid map
    :param tet_surface: tid to contained surface fid map
    :param surface_tet_faces: surface faces in tet vid
    :param winding numbers: winding numbers for input tets
    """

    # construct TV and VT connectivity
    TV = tets.tolist()
    TV_preserved = [
        True for i in range(tets.shape[0])
    ]  # flag marking tet is preserved (True) or removed (False)
    VT = [[] for i in range(vertices.shape[0])]
    V = copy.deepcopy(vertices)

    for tid, tet in enumerate(tets):
        # print(tet)
        for vid in tet:
            # print(vid)
            VT[vid].append(tid)

    # print(VT)
    # print(len(VT))
    # exit(0)

    # convert from dict to list
    # new_winding_numbers = copy.deepcopy(winding_numbers)
    new_winding_numbers = []
    for i in range(tets.shape[0]):
        new_winding_numbers.append(winding_numbers[i])

    # get edges on the surface in tid. computed both order tuple v0v1 v1v0 = True
    edge_on_surface = {}
    for f in surface_tet_faces:
        edge_on_surface[str(f[0]) + "+" + str(f[1])] = True
        edge_on_surface[str(f[1]) + "+" + str(f[0])] = True
        edge_on_surface[str(f[0]) + "+" + str(f[2])] = True
        edge_on_surface[str(f[2]) + "+" + str(f[0])] = True
        edge_on_surface[str(f[1]) + "+" + str(f[2])] = True
        edge_on_surface[str(f[2]) + "+" + str(f[1])] = True

    # get faces on the surface in tid. computed in sorted order. map[(f0,f1,f2)] to fid
    face_on_surface = {}
    for i, f in enumerate(surface_tet_faces):
        f_sorted = sorted(f)
        face_on_surface[
            str(f_sorted[0]) + "+" + str(f_sorted[1]) + "+" + str(f_sorted[2])
        ] = i

    # step 1, split all tets has > 1 boundary edges by adding a barycenter
    # iterate through original tets
    original_tet_cnt = tets.shape[0]

    for i in range(original_tet_cnt):
        tet = TV[i]
        tet_edges = [
            str(tet[0]) + "+" + str(tet[1]),
            str(tet[0]) + "+" + str(tet[2]),
            str(tet[0]) + "+" + str(tet[3]),
            str(tet[1]) + "+" + str(tet[2]),
            str(tet[1]) + "+" + str(tet[3]),
            str(tet[2]) + "+" + str(tet[3]),
        ]

        # check how many boundary edges
        boundary_edge_cnt = 0
        for e in tet_edges:
            if e in edge_on_surface:
                boundary_edge_cnt += 1
                if boundary_edge_cnt > 1:
                    break

        if boundary_edge_cnt <= 1:
            # not tet of interest
            continue
        else:
            # add barycenter to VT
            new_vid = V.shape[0]
            new_v_pos = (V[tet[0]] + V[tet[1]] + V[tet[2]] + V[tet[3]]) / 4.0

            V = np.vstack([V, new_v_pos])  # add new vertex to V

            # remove old tet
            TV_preserved[i] = False  # remove by mark as false

            # add new tets
            new_tids = list(range(len(TV), len(TV) + 4))
            TV_preserved += [True, True, True, True]  # add four new tets
            TV.append([new_vid, tet[1], tet[2], tet[3]])
            TV.append([tet[0], new_vid, tet[2], tet[3]])
            TV.append([tet[0], tet[1], new_vid, tet[3]])
            TV.append([tet[0], tet[1], tet[2], new_vid])
            new_winding_numbers += [
                new_winding_numbers[i],
                new_winding_numbers[i],
                new_winding_numbers[i],
                new_winding_numbers[i],
            ]  # add four winding numbers, same as tet i

            # add new vertex to VT
            VT.append(new_tids)

            # delete tet i from VT
            for tvid in tet:
                VT[tvid].remove(i)

            # add new tets to VT
            VT[tet[0]] += [new_tids[1], new_tids[2], new_tids[3]]
            VT[tet[1]] += [new_tids[0], new_tids[2], new_tids[3]]
            VT[tet[2]] += [new_tids[0], new_tids[1], new_tids[3]]
            VT[tet[3]] += [new_tids[0], new_tids[1], new_tids[2]]

    # now all tets has at most 3 boundary edges, and they are in the same tet face

    # step 2, for all the new sub tets, split the face f into 3 if f has > 1 boundary edge.
    # This affects the other tet sharing f
    # no need to split if f is exactly a face on the boundary

    tet_cnt_after_step_1 = len(TV)

    # iterate through new tets
    for i in range(original_tet_cnt, tet_cnt_after_step_1):
        if not TV_preserved[i]:
            # skip tet that is splitted
            continue

        tet = TV[i]
        tet_edges = [
            [tet[0], tet[1]],
            [tet[0], tet[2]],
            [tet[0], tet[3]],
            [tet[1], tet[2]],
            [tet[1], tet[3]],
            [tet[2], tet[3]],
        ]
        boundary_edge_vertices = []

        # check how many boundary edges
        boundary_edge_cnt = 0
        for e in tet_edges:
            if str(e[0]) + "+" + str(e[1]) in edge_on_surface:
                boundary_edge_cnt += 1
                boundary_edge_vertices.append(e[0])
                boundary_edge_vertices.append(e[1])
        assert boundary_edge_cnt <= 3

        if boundary_edge_cnt <= 1:
            # not tet of interest
            continue

        bev_unique = sorted(list(set(boundary_edge_vertices)))
        assert len(bev_unique) == 3

        if boundary_edge_cnt == 3:
            # check if all three edges forms a boundary face
            if (
                str(bev_unique[0]) + "+" + str(bev_unique[1]) + "+" + str(bev_unique[2])
                in face_on_surface
            ):
                # this is a tet with a boundary face, skip
                continue

        # split the face into 3 that contains the boundary edges
        # get the tets incident to the face to split

        incident_tets = list(
            set(VT[bev_unique[0]]) & set(VT[bev_unique[1]]) & set(VT[bev_unique[2]])
        )

        assert len(incident_tets) > 0
        if len(incident_tets) > 2:
            print(bev_unique)
            # print(incident_tets)
            print(len(incident_tets))
        assert len(incident_tets) <= 2
        assert i in incident_tets

        # add new vertex (face barycenter)
        new_vid = V.shape[0]
        new_v_pos = (V[bev_unique[0]] + V[bev_unique[1]] + V[bev_unique[2]]) / 3.0
        V = np.vstack([V, new_v_pos])  # add new vertex to V
        VT.append([])  # add slot to VT

        for tid in incident_tets:
            incident_tet = TV[tid]
            bev_local_ids = [-1, -1, -1]
            for k in range(4):
                for j in range(3):
                    if incident_tet[k] == bev_unique[j]:
                        bev_local_ids[j] = k
            assert all(lid > -1 for lid in bev_local_ids)

            apex_vid = -1
            for k in range(4):
                if k not in bev_local_ids:
                    apex_vid = incident_tet[k]
                    break
            assert apex_vid > -1

            # remove current tet
            TV_preserved[tid] = False

            # add 3 new tets
            new_tids = list(range(len(TV), len(TV) + 3))
            TV_preserved += [True, True, True]

            new_tet_0 = [
                incident_tet[0],
                incident_tet[1],
                incident_tet[2],
                incident_tet[3],
            ]
            new_tet_0[bev_local_ids[0]] = (
                new_vid  # replace the boundary vertex local id 0 as new vid
            )

            new_tet_1 = [
                incident_tet[0],
                incident_tet[1],
                incident_tet[2],
                incident_tet[3],
            ]
            new_tet_1[bev_local_ids[1]] = (
                new_vid  # replace the boundary vertex local id 1 as new vid
            )

            new_tet_2 = [
                incident_tet[0],
                incident_tet[1],
                incident_tet[2],
                incident_tet[3],
            ]
            new_tet_2[bev_local_ids[2]] = (
                new_vid  # replace the boundary vertex local id 2 as new vid
            )

            TV.append(new_tet_0)
            TV.append(new_tet_1)
            TV.append(new_tet_2)

            new_winding_numbers += [
                new_winding_numbers[tid],
                new_winding_numbers[tid],
                new_winding_numbers[tid],
            ]  # add three new winding numbers

            # add new tets to new vid VT
            VT[new_vid] += new_tids

            # delete tet tid from old VT
            for tvid in incident_tet:
                # print(tvid)
                # print(VT[tvid], tid)
                VT[tvid].remove(tid)

            # add new tets to old VT
            VT[apex_vid] += new_tids  # add 3 to apex
            VT[incident_tet[bev_local_ids[0]]] += [new_tids[1], new_tids[2]]
            VT[incident_tet[bev_local_ids[1]]] += [new_tids[0], new_tids[2]]
            VT[incident_tet[bev_local_ids[2]]] += [new_tids[0], new_tids[1]]

    # finalize, remove removed tets
    final_tets = []
    final_winding_numbers = []
    for i in range(len(TV)):
        if TV_preserved[i]:
            final_tets.append(TV[i])
            final_winding_numbers.append(new_winding_numbers[i])

    final_tets = np.array(final_tets)
    final_vertices = V

    # build surface_adj_tet and tet_surface
    new_surface_adj_tet = {}
    new_tet_surface = {}

    for tid, tet in enumerate(final_tets):
        tet_faces = [
            sorted([tet[0], tet[1], tet[2]]),
            sorted([tet[0], tet[1], tet[3]]),
            sorted([tet[0], tet[2], tet[3]]),
            sorted([tet[1], tet[2], tet[3]]),
        ]

        for f in tet_faces:
            f_str = str(f[0]) + "+" + str(f[1]) + "+" + str(f[2])
            if f_str in face_on_surface:
                fid = face_on_surface[f_str]

                # update new surface_adj_tet
                if fid not in new_surface_adj_tet:
                    new_surface_adj_tet[fid] = [tid]
                else:
                    new_surface_adj_tet[fid].append(tid)
                    assert len(new_surface_adj_tet[fid]) <= 2

                # update new tet_surface
                assert (
                    tid not in new_tet_surface
                )  # each tet can only have one face on boundary
                new_tet_surface[tid] = [fid]

    # TODO: check orientation

    print("[{}] ".format(datetime.datetime.now()), "Done Isolate surface edges/faces.")

    return (
        final_tets,
        final_vertices,
        new_surface_adj_tet,
        new_tet_surface,
        final_winding_numbers,
    )


def check_surface_edges_isolation(tets, tet_vertices, surface_tet_faces):
    edge_on_surface = {}
    face_on_surface = {}

    for i, f in enumerate(surface_tet_faces):
        edge_on_surface[str(f[0]) + "+" + str(f[1])] = True
        edge_on_surface[str(f[1]) + "+" + str(f[0])] = True
        edge_on_surface[str(f[0]) + "+" + str(f[2])] = True
        edge_on_surface[str(f[2]) + "+" + str(f[0])] = True
        edge_on_surface[str(f[1]) + "+" + str(f[2])] = True
        edge_on_surface[str(f[2]) + "+" + str(f[1])] = True

        f_sorted = sorted(f)

        face_on_surface[
            str(f_sorted[0]) + "+" + str(f_sorted[1]) + "+" + str(f_sorted[2])
        ] = i

    for tet in tets:
        tet_edges = [
            str(tet[0]) + "+" + str(tet[1]),
            str(tet[0]) + "+" + str(tet[2]),
            str(tet[0]) + "+" + str(tet[3]),
            str(tet[1]) + "+" + str(tet[2]),
            str(tet[1]) + "+" + str(tet[3]),
            str(tet[2]) + "+" + str(tet[3]),
        ]

        tet_faces = [
            [tet[0], tet[1], tet[2]],
            [tet[0], tet[1], tet[3]],
            [tet[0], tet[2], tet[3]],
            [tet[1], tet[2], tet[3]],
        ]

        surface_edge_cnt = 0
        surface_face_cnt = 0

        for e in tet_edges:
            if e in edge_on_surface:
                surface_edge_cnt += 1

        for f in tet_faces:
            f_sorted = sorted(f)
            if (
                str(f_sorted[0]) + "+" + str(f_sorted[1]) + "+" + str(f_sorted[2])
                in face_on_surface
            ):
                surface_face_cnt += 1

        if surface_face_cnt == 1:
            assert surface_edge_cnt == 3
        elif surface_face_cnt > 1:
            assert False
        else:
            assert surface_edge_cnt <= 1
