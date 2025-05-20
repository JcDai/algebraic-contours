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
        print("[{}] ".format(datetime.datetime.now()),
              "checking tet orientation ...")
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
        tet_indices_touching_surface = np.unique(np.argwhere(
            np.isin(tets_unsliced, surface_vertices))[:, 0])

        tets = tets_unsliced[tet_indices_touching_surface]
        winding_numbers_data = winding_numbers_data_unsliced[tet_indices_touching_surface]
        vertices = vertices_unsliced
        vertices, tets, _, sliced_to_unsliced_v_map = igl.remove_unreferenced(
            vertices_unsliced, tets)

        # print(winding_numbers_data.shape)
        # print(winding_numbers_data)
        # print(tets.shape)

        # fix  winding number shape
        if len(winding_numbers_data.shape) == 1:
            winding_numbers_data = winding_numbers_data[:, None]
        # print(winding_numbers_data.shape)
        # print(winding_numbers_data)

        # exit()

        m_sliced = mio.Mesh(vertices, [('tetra', tets)], cell_data={
                            "winding_number": winding_numbers_data.T})
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
                surface_tet_faces[i][j] = unsliced_to_sliced_v_map[surface_tet_faces[i][j]]
        surface_tet_faces = np.array(surface_tet_faces)
        # print(surface_tet_faces)

    para_in_v, para_in_f, im, para_in_v_to_tet_v_map = igl.remove_unreferenced(
        vertices, surface_tet_faces
    )
    assert (igl.bfs_orient(para_in_f)[0] == para_in_f).all()

    igl.write_obj(workspace_path + "embedded_surface.obj",
                  para_in_v, para_in_f)
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

    return tets, vertices, winding_numbers, tet_surface_origin, surface_adj_tet, para_in_v, para_in_f, para_in_v_to_tet_v_map, surface_tet_faces, surface_vertices


# simplicial embedding
def simplicial_embedding(tets, vertices, winding_numbers, tet_surface_origin, surface_adj_tet, surface_tet_faces):
    print("[{}] ".format(datetime.datetime.now()),
          "Doing simplicial embedding ...")
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
        "Done simplicial embedding. Splitted {} tets".format(
            simplicial_embedding_cnt),
    )

    return tets_regular, tets_vertices_regular, surface_adj_tet, tet_surface, winding_numbers
