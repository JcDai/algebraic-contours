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


def build_bezier_hard_constraint_matrix(workspace_path, tri_to_tet_index_mapping_file, bezier_cons_mat_file, linear_tetmesh_file):
    print(
        "[{}] ".format(datetime.datetime.now()),
        "constructing full bezier hard constraint matrix ...",
    )

    bezier_cons_matrix = scipy.io.mmread(bezier_cons_mat_file)
    local2global = np.loadtxt(
        workspace_path + tri_to_tet_index_mapping_file
    ).astype(np.int32)

    A = bezier_cons_matrix.tocoo(True)
    m = mio.read(workspace_path + linear_tetmesh_file)
    v = m.points
    b = -(A @ v[local2global, :])

    with h5py.File(workspace_path + "CT_bezier_constraint_matrix.hdf5", "w") as f:
        f.create_dataset("local2global", data=local2global.astype(np.int32))
        f.create_dataset("A_triplets/values", data=A.data)
        f.create_dataset("A_triplets/cols", data=A.col)
        f.create_dataset("A_triplets/rows", data=A.row)
        f.create_dataset("A_triplets/shape", data=A.shape)
        f.create_dataset("b", data=b)

    local2global_matrix_rows = [i for i in range(local2global.shape[0])]
    local2global_matrix_cols = [local2global[i]
                                for i in range(local2global.shape[0])]
    local2global_matrix_data = [1.0] * local2global.shape[0]

    with h5py.File(workspace_path + "local2global_matrix.hdf5", "w") as f:
        f.create_dataset(
            "weight_triplets/values", data=np.array(local2global_matrix_data)
        )
        f.create_dataset(
            "weight_triplets/cols",
            data=np.array(local2global_matrix_cols).astype(np.int32),
        )
        f.create_dataset(
            "weight_triplets/rows",
            data=np.array(local2global_matrix_rows).astype(np.int32),
        )
        f["weight_triplets"].attrs["shape"] = (
            local2global.shape[0], v.shape[0])


def build_bezier_reduce2full_matrix(workspace_path, tri_to_tet_index_mapping_file, bezier_reduced2full_file, bezier_r2f_col_idx_map_file):
    print(
        "[{}] ".format(datetime.datetime.now()),
        "constructing bezier reduced to full matrix ...",
    )

    bezier_r2f = scipy.io.mmread(bezier_reduced2full_file)
    local2global = np.loadtxt(
        workspace_path + tri_to_tet_index_mapping_file
    ).astype(np.int32)

    col2local = np.loadtxt(
        workspace_path + bezier_r2f_col_idx_map_file
    ).astype(np.int32)

    col2global = local2global[col2local]
    print(col2global.shape)
    print(col2local.shape)

    A = bezier_r2f.tocoo(True)

    with h5py.File(workspace_path + "CT_bezier_r2f_matrix.hdf5", "w") as f:
        f.create_dataset("local2global", data=col2global.astype(np.int32))
        f.create_dataset("A_triplets/values", data=A.data)
        f.create_dataset("A_triplets/cols", data=A.col)
        f.create_dataset("A_triplets/rows", data=A.row)
        f.create_dataset("A_triplets/shape", data=A.shape)


def build_expanded_bezier_hard_constraint_matrix(workspace_path, tri_to_tet_index_mapping_file, bezier_cons_mat_file, linear_tetmesh_file, tet_edge_to_vertices, tet_face_to_vertices, bezier_reduced2full_file, bezier_r2f_col_idx_map_file):
    print(
        "[{}] ".format(datetime.datetime.now()),
        "constructing expanded full bezier hard constraint matrix ...",
    )

    m = mio.read(workspace_path + linear_tetmesh_file)
    v = m.points

    # compute nodes on boundaries
    cells_high_order = m.cells_dict["tetra20"]
    cells = cells_high_order[:, :4]
    bd_f = igl.boundary_facets(cells)
    bd_v = []
    for f in bd_f:
        # corners
        for i in range(3):
            bd_v.append(f[i])

        # edges
        e01 = str(f[0]) + "+" + str(f[1])
        e12 = str(f[1]) + "+" + str(f[2])
        e20 = str(f[2]) + "+" + str(f[0])
        bd_v.extend(tet_edge_to_vertices[e01])
        bd_v.extend(tet_edge_to_vertices[e12])
        bd_v.extend(tet_edge_to_vertices[e20])

        # faces
        f_sorted = [f[0], f[1], f[2]]
        f_sorted.sort()
        f_str = str(f_sorted[0]) + "+" + \
            str(f_sorted[1]) + "+" + str(f_sorted[2])
        bd_v.append(tet_face_to_vertices[f_str])

    bd_v = np.unique(np.array(bd_v))

    # add expanded c1 constraint
    bezier_cons_matrix = scipy.io.mmread(bezier_cons_mat_file)
    local2global = np.loadtxt(
        workspace_path + tri_to_tet_index_mapping_file
    ).astype(np.int32)

    A_c1 = bezier_cons_matrix.tocoo(True)

    A_c1_row = A_c1.row
    A_c1_col = A_c1.col
    A_c1_data = A_c1.data

    A_c1_tet_row = A_c1_row
    A_c1_tet_col = local2global[A_c1_col]
    A_c1_tet_data = A_c1_data

    A_c1_tet_expanded_row = [0 for i in range(A_c1_tet_row.shape[0]*3)]
    A_c1_tet_expanded_col = [0 for i in range(A_c1_tet_col.shape[0]*3)]
    A_c1_tet_expanded_data = [0 for i in range(A_c1_tet_data.shape[0]*3)]

    for i in range(A_c1_tet_row.shape[0]):
        A_c1_tet_expanded_row[i*3+0] = A_c1_tet_row[i]*3+0
        A_c1_tet_expanded_row[i*3+1] = A_c1_tet_row[i]*3+1
        A_c1_tet_expanded_row[i*3+2] = A_c1_tet_row[i]*3+2

        A_c1_tet_expanded_col[i*3+0] = A_c1_tet_col[i]*3+0
        A_c1_tet_expanded_col[i*3+1] = A_c1_tet_col[i]*3+1
        A_c1_tet_expanded_col[i*3+2] = A_c1_tet_col[i]*3+2

        A_c1_tet_expanded_data[i*3+0] = A_c1_tet_data[i]
        A_c1_tet_expanded_data[i*3+1] = A_c1_tet_data[i]
        A_c1_tet_expanded_data[i*3+2] = A_c1_tet_data[i]

    A_c1_tet_expanded = scipy.sparse.coo_array((A_c1_tet_expanded_data, (
        A_c1_tet_expanded_row, A_c1_tet_expanded_col)), shape=(A_c1.shape[0]*3, v.shape[0]*3))

    # add dirichlet
    dirichlet_row = np.array([i for i in range(bd_v.shape[0])])
    dirichlet_col = np.array([id for id in bd_v])
    dirichlet_data = np.array([1 for i in range(bd_v.shape[0])])

    dirichlet = scipy.sparse.coo_array(
        (dirichlet_data, (dirichlet_row, dirichlet_col)), shape=(bd_v.shape[0], v.shape[0]))

    dirichlet_expanded_row = [0 for i in range(dirichlet_row.shape[0]*3)]
    dirichlet_expanded_col = [0 for i in range(dirichlet_col.shape[0]*3)]
    dirichlet_expanded_data = [0 for i in range(dirichlet_data.shape[0]*3)]

    for i in range(dirichlet_row.shape[0]):
        dirichlet_expanded_row[i*3+0] = dirichlet_row[i]*3+0
        dirichlet_expanded_row[i*3+1] = dirichlet_row[i]*3+1
        dirichlet_expanded_row[i*3+2] = dirichlet_row[i]*3+2

        dirichlet_expanded_col[i*3+0] = dirichlet_col[i]*3+0
        dirichlet_expanded_col[i*3+1] = dirichlet_col[i]*3+1
        dirichlet_expanded_col[i*3+2] = dirichlet_col[i]*3+2

        dirichlet_expanded_data[i*3+0] = dirichlet_data[i]
        dirichlet_expanded_data[i*3+1] = dirichlet_data[i]
        dirichlet_expanded_data[i*3+2] = dirichlet_data[i]

    dirichlet_expanded = scipy.sparse.coo_array((dirichlet_expanded_data, (
        dirichlet_expanded_row, dirichlet_expanded_col)), shape=(dirichlet.shape[0]*3, dirichlet.shape[1]*3))

    stacked = scipy.sparse.vstack((A_c1_tet_expanded, dirichlet_expanded))

    # compute b
    v_expanded = np.reshape(v, (v.shape[0]*3, 1)).T[0]
    b_c1_expanded = -(A_c1_tet_expanded @ v_expanded)

    b_dirichlet = np.array([0 for i in range(dirichlet_expanded.shape[0])])

    b = np.concatenate((b_c1_expanded, b_dirichlet))
    print("b shape: ", b.shape)

    with h5py.File(workspace_path + "CT_bezier_constraint_matrix_with_dirichlet_expanded.hdf5", "w") as f:
        f.create_dataset("A_triplets/values", data=stacked.data)
        f.create_dataset("A_triplets/cols", data=stacked.col)
        f.create_dataset("A_triplets/rows", data=stacked.row)
        f.create_dataset("A_triplets/shape", data=stacked.shape)
        # R2F_triplets
        # proj
        f.create_dataset("b", data=b)

    #####################################
    ######## reduce to full #############
    #####################################

    bezier_r2f = scipy.io.mmread(bezier_reduced2full_file)
    local2global = np.loadtxt(
        workspace_path + tri_to_tet_index_mapping_file
    ).astype(np.int32)

    col2local = np.loadtxt(
        workspace_path + bezier_r2f_col_idx_map_file
    ).astype(np.int32)

    col2global = local2global[col2local]

    # compute independent node ids
    sf_ind_node_ids = col2global.astype(int)
    sf_dep_node_ids = []
    for id in local2global:
        if id not in sf_ind_node_ids:
            sf_dep_node_ids.append(id)
    sf_dep_node_ids = np.array(sf_dep_node_ids)

    other_dep_node_ids = bd_v
    other_ind_node_ids = []
    for id in range(v.shape[0]):
        if id not in local2global and id not in bd_v:
            other_ind_node_ids.append(id)
    other_ind_node_ids = np.array(other_ind_node_ids)

    assert (sf_ind_node_ids.shape[0] + sf_dep_node_ids.shape[0] +
            other_dep_node_ids.shape[0] + other_ind_node_ids.shape[0] == v.shape[0])

    # compute full to reduce node id mapping
    all_ind_node_ids = np.sort(np.concatenate(
        (sf_ind_node_ids, other_ind_node_ids)))

    # print(all_ind_node_ids)

    global2ind = {}
    for i in range(all_ind_node_ids.shape[0]):
        global2ind[all_ind_node_ids[i]] = i

    # build r2f

    bezier_r2f_row = bezier_r2f.row
    bezier_r2f_col = bezier_r2f.col
    bezier_r2f_data = bezier_r2f.data

    r2f_full_row = []
    r2f_full_col = []
    r2f_full_data = []

    # add sf r2f
    for i in range(bezier_r2f_row.shape[0]):
        r2f_full_row.append(local2global[bezier_r2f_row[i]])
        r2f_full_col.append(global2ind[col2global[bezier_r2f_col[i]]])
        r2f_full_data.append(bezier_r2f_data[i])

    # add other r2f
    for i in range(other_ind_node_ids.shape[0]):
        r2f_full_row.append(other_ind_node_ids[i])
        r2f_full_col.append(global2ind[other_ind_node_ids[i]])
        r2f_full_data.append(1)

    r2f_full = scipy.sparse.coo_array((r2f_full_data, (r2f_full_row, r2f_full_col)), shape=(
        v.shape[0], all_ind_node_ids.shape[0]))

    print(r2f_full.shape)
    # print(bd_v.shape[0])
    # print(local2global.shape[0]-col2global.shape[0])

    # expand r2f_full
    r2f_full_expanded_row = [0 for i in range(len(r2f_full_row) * 3)]
    r2f_full_expanded_col = [0 for i in range(len(r2f_full_col) * 3)]
    r2f_full_expanded_data = [0 for i in range(len(r2f_full_data) * 3)]

    for i in range(len(r2f_full_row)):
        r2f_full_expanded_row[i*3+0] = r2f_full_row[i]*3+0
        r2f_full_expanded_row[i*3+1] = r2f_full_row[i]*3+1
        r2f_full_expanded_row[i*3+2] = r2f_full_row[i]*3+2

        r2f_full_expanded_col[i*3+0] = r2f_full_col[i]*3+0
        r2f_full_expanded_col[i*3+1] = r2f_full_col[i]*3+1
        r2f_full_expanded_col[i*3+2] = r2f_full_col[i]*3+2

        r2f_full_expanded_data[i*3+0] = r2f_full_data[i]
        r2f_full_expanded_data[i*3+1] = r2f_full_data[i]
        r2f_full_expanded_data[i*3+2] = r2f_full_data[i]

    r2f_full_expanded = scipy.sparse.coo_array((r2f_full_expanded_data, (r2f_full_expanded_row, r2f_full_expanded_col)), shape=(
        r2f_full.shape[0]*3, r2f_full.shape[1]*3))

    print(r2f_full_expanded.shape)

    with h5py.File(workspace_path + "CT_bezier_r2f_with_dirichlet_expanded.hdf5", "w") as f:
        f.create_dataset("A_triplets/values", data=r2f_full_expanded.data)
        f.create_dataset("A_triplets/cols", data=r2f_full_expanded.col)
        f.create_dataset("A_triplets/rows", data=r2f_full_expanded.row)
        f.create_dataset("A_triplets/shape", data=r2f_full_expanded.shape)
        # r2f triplets

    # compute proj
    v_reduced = v[all_ind_node_ids]
    v_expanded_reduce = np.reshape(v_reduced, (v_reduced.shape[0]*3, 1)).T[0]
    print(v_expanded_reduce.shape)
    print(r2f_full_expanded.shape)

    proj = r2f_full_expanded @ v_expanded_reduce - v_expanded
    for id in other_dep_node_ids:
        proj[id*3] = 0
        proj[id*3 + 1] = 0
        proj[id*3 + 2] = 0
    for id in other_ind_node_ids:
        proj[id*3] = 0
        proj[id*3 + 1] = 0
        proj[id*3 + 2] = 0

    test_vec = np.random.rand(v_expanded_reduce.shape[0])

    full = (r2f_full_expanded @ test_vec + proj)

    error = stacked @ full - b
    print("error: ", np.linalg.norm(error))

    print(other_dep_node_ids)

    # print(np.linalg.norm(error[other_dep_node_ids]))

    nn = other_dep_node_ids.shape[0]
    # print(np.linalg.norm(error[(b.shape[0]-nn+1):-1]))
    print(np.linalg.norm(b[(b.shape[0]-nn):-1]))

    # print(np.linalg.norm(b[other_dep_node_ids]))
    # print(np.linalg.norm(b[other_ind_node_ids]))

    # print("kk")
    # print(np.min(r2f_full_expanded.todense()[other_dep_node_ids * 3, :]))
    # print(np.max(r2f_full_expanded.todense()[other_dep_node_ids * 3 + 1, :]))
    # print(np.min(r2f_full_expanded.todense()[other_dep_node_ids * 3 + 2, :]))
    # print(np.max(r2f_full_expanded.todense()[other_dep_node_ids * 3, :]))
    # print(np.min(r2f_full_expanded.todense()[other_dep_node_ids * 3 + 1, :]))
    # print(np.max(r2f_full_expanded.todense()[other_dep_node_ids * 3 + 2, :]))

    with h5py.File(workspace_path + "CT_bezier_all_matrices.hdf5", "w") as f:
        f.create_dataset("A_triplets/values", data=stacked.data)
        f.create_dataset("A_triplets/cols", data=stacked.col.astype(np.int32))
        f.create_dataset("A_triplets/rows", data=stacked.row.astype(np.int32))
        f.create_dataset("A_triplets/shape", data=stacked.shape)

        f.create_dataset("A_proj_triplets/values", data=r2f_full_expanded.data)
        f.create_dataset("A_proj_triplets/cols",
                         data=r2f_full_expanded.col.astype(np.int32))
        f.create_dataset("A_proj_triplets/rows",
                         data=r2f_full_expanded.row.astype(np.int32))
        f.create_dataset("A_proj_triplets/shape", data=r2f_full_expanded.shape)

        f.create_dataset("b_proj", data=proj[:, None])

        f.create_dataset("b", data=b[:, None])


def build_full_expanded_bezier_hard_constraint_matrix(workspace_path, tri_to_tet_index_mapping_file, bezier_cons_mat_file, linear_tetmesh_file, tet_edge_to_vertices, tet_face_to_vertices, bezier_reduced2full_file, bezier_r2f_col_idx_map_file, initial_interp_mesh):
    print(
        "[{}] ".format(datetime.datetime.now()),
        "constructing expanded full (including cone) bezier hard constraint matrix ...",
    )

    m = mio.read(workspace_path + linear_tetmesh_file)
    v = m.points

    # check mesh orientation
    # tt = m.cells_dict['tetra20']
    # oris = []
    # for tet in tt:
    #     ori = orient3d(v[tet[0]], v[tet[1]], v[tet[2]], v[tet[3]])
    #     oris.append(ori)
    #     # print(ori)
    # for ori in oris:
    #     if ori:
    #         print("has positive one")

    # compute nodes on boundaries
    cells_high_order = m.cells_dict["tetra20"]
    cells = cells_high_order[:, :4]
    bd_f = igl.boundary_facets(cells)
    bd_v = []
    for f in bd_f:
        # corners
        for i in range(3):
            bd_v.append(f[i])

        # edges
        e01 = str(f[0]) + "+" + str(f[1])
        e12 = str(f[1]) + "+" + str(f[2])
        e20 = str(f[2]) + "+" + str(f[0])
        bd_v.extend(tet_edge_to_vertices[e01])
        bd_v.extend(tet_edge_to_vertices[e12])
        bd_v.extend(tet_edge_to_vertices[e20])

        # faces
        f_sorted = [f[0], f[1], f[2]]
        f_sorted.sort()
        f_str = str(f_sorted[0]) + "+" + \
            str(f_sorted[1]) + "+" + str(f_sorted[2])
        bd_v.append(tet_face_to_vertices[f_str])

    bd_v = np.unique(np.array(bd_v))

    # add expanded c1 constraint
    bezier_cons_matrix = scipy.io.mmread(
        bezier_cons_mat_file)  # this is already expanded to xyz
    local2global = np.loadtxt(
        workspace_path + tri_to_tet_index_mapping_file
    ).astype(np.int32)

    local2global_expanded = []
    for i in range(local2global.shape[0]):
        local2global_expanded.append(local2global[i] * 3 + 0)
        local2global_expanded.append(local2global[i] * 3 + 1)
        local2global_expanded.append(local2global[i] * 3 + 2)
    local2global_expanded = np.array(local2global_expanded)

    assert bezier_cons_matrix.shape[1] == 3 * local2global.shape[0]

    A_c1 = bezier_cons_matrix.tocoo(True)

    A_c1_row = A_c1.row
    A_c1_col = A_c1.col
    A_c1_data = A_c1.data

    A_c1_tet_expanded_row = A_c1_row
    A_c1_tet_expanded_col = local2global_expanded[A_c1_col]
    A_c1_tet_expanded_data = A_c1_data

    A_c1_tet_expanded = scipy.sparse.coo_array((A_c1_tet_expanded_data, (
        A_c1_tet_expanded_row, A_c1_tet_expanded_col)), shape=(A_c1.shape[0]*3, v.shape[0]*3))

    # add dirichlet
    dirichlet_row = np.array([i for i in range(bd_v.shape[0])])
    dirichlet_col = np.array([id for id in bd_v])
    dirichlet_data = np.array([1 for i in range(bd_v.shape[0])])

    dirichlet = scipy.sparse.coo_array(
        (dirichlet_data, (dirichlet_row, dirichlet_col)), shape=(bd_v.shape[0], v.shape[0]))

    dirichlet_expanded_row = [0 for i in range(dirichlet_row.shape[0]*3)]
    dirichlet_expanded_col = [0 for i in range(dirichlet_col.shape[0]*3)]
    dirichlet_expanded_data = [0 for i in range(dirichlet_data.shape[0]*3)]

    for i in range(dirichlet_row.shape[0]):
        dirichlet_expanded_row[i*3+0] = dirichlet_row[i]*3+0
        dirichlet_expanded_row[i*3+1] = dirichlet_row[i]*3+1
        dirichlet_expanded_row[i*3+2] = dirichlet_row[i]*3+2

        dirichlet_expanded_col[i*3+0] = dirichlet_col[i]*3+0
        dirichlet_expanded_col[i*3+1] = dirichlet_col[i]*3+1
        dirichlet_expanded_col[i*3+2] = dirichlet_col[i]*3+2

        dirichlet_expanded_data[i*3+0] = dirichlet_data[i]
        dirichlet_expanded_data[i*3+1] = dirichlet_data[i]
        dirichlet_expanded_data[i*3+2] = dirichlet_data[i]

    dirichlet_expanded = scipy.sparse.coo_array((dirichlet_expanded_data, (
        dirichlet_expanded_row, dirichlet_expanded_col)), shape=(dirichlet.shape[0]*3, dirichlet.shape[1]*3))

    stacked = scipy.sparse.vstack((A_c1_tet_expanded, dirichlet_expanded))

    # compute b
    v_expanded = np.reshape(v, (v.shape[0]*3, 1)).T[0]
    b_c1_expanded = -(A_c1_tet_expanded @ v_expanded)

    b_dirichlet = np.array([0 for i in range(dirichlet_expanded.shape[0])])

    b = np.concatenate((b_c1_expanded, b_dirichlet))
    print("b shape: ", b.shape)

    with h5py.File(workspace_path + "CT_bezier_constraint_matrix_with_dirichlet_expanded.hdf5", "w") as f:
        f.create_dataset("A_triplets/values", data=stacked.data)
        f.create_dataset("A_triplets/cols", data=stacked.col)
        f.create_dataset("A_triplets/rows", data=stacked.row)
        f.create_dataset("A_triplets/shape", data=stacked.shape)
        # R2F_triplets
        # proj
        f.create_dataset("b", data=b)

    #####################################
    ######## reduce to full #############
    #####################################

    bezier_r2f = scipy.io.mmread(bezier_reduced2full_file)  # already expanded
    print("bezier r2f shape: ", bezier_r2f.shape)

    col2local_expanded = np.loadtxt(
        workspace_path + bezier_r2f_col_idx_map_file
    ).astype(np.int32)  # already expanded

    col2global_expanded = local2global_expanded[col2local_expanded]
    print("here1")

    # compute independent node ids
    sf_ind_ids = col2global_expanded.astype(int)  # expanded
    sf_dep_ids = []  # expanded

    sf_ind_ids_dict = {}
    for id in sf_ind_ids:
        sf_ind_ids_dict[id] = True

    for id in local2global_expanded:
        if id not in sf_ind_ids_dict:
            sf_dep_ids.append(id)
    sf_dep_ids = np.array(sf_dep_ids)
    print("here2")

    other_dep_ids_unexpanded = bd_v  # not expanded
    other_dep_ids = []  # expanded
    for i in range(bd_v.shape[0]):
        other_dep_ids.append(other_dep_ids_unexpanded[i] * 3 + 0)
        other_dep_ids.append(other_dep_ids_unexpanded[i] * 3 + 1)
        other_dep_ids.append(other_dep_ids_unexpanded[i] * 3 + 2)
    other_dep_ids = np.array(other_dep_ids)

    other_dep_ids_dict = {}
    for id in other_dep_ids:
        other_dep_ids_dict[id] = True

    local2global_expanded_dict = {}
    for id in local2global_expanded:
        local2global_expanded_dict[id] = True

    other_ind_ids = []  # expanded/ vertex xyz not on surface or boundary
    for id in range(v.shape[0]*3):
        if id not in local2global_expanded_dict and id not in other_dep_ids_dict:
            other_ind_ids.append(id)
    other_ind_ids = np.array(other_ind_ids)

    assert sf_ind_ids.shape[0] + sf_dep_ids.shape[0] + \
        other_dep_ids.shape[0] + other_ind_ids.shape[0] == v.shape[0] * 3

    print("here3")

    # compute full to reduce id mapping
    all_ind_ids = np.sort(np.concatenate((sf_ind_ids, other_ind_ids)))

    print("here4")

    global2ind = {}  # expanded
    for i in range(all_ind_ids.shape[0]):
        global2ind[all_ind_ids[i]] = i

    # build r2f
    bezier_r2f_row = bezier_r2f.row
    bezier_r2f_col = bezier_r2f.col
    bezier_r2f_data = bezier_r2f.data

    r2f_full_row = []
    r2f_full_col = []
    r2f_full_data = []

    # add sf r2f
    for i in range(bezier_r2f_row.shape[0]):
        r2f_full_row.append(local2global_expanded[bezier_r2f_row[i]])
        r2f_full_col.append(global2ind[col2global_expanded[bezier_r2f_col[i]]])
        r2f_full_data.append(bezier_r2f_data[i])

    # add other r2f
    for i in range(other_ind_ids.shape[0]):
        r2f_full_row.append(other_ind_ids[i])
        r2f_full_col.append(global2ind[other_ind_ids[i]])
        r2f_full_data.append(1)

    # TODO: check this! are boundary treated as indepdent?
    # for i in range(other_dep_ids.shape[0]):
    #     r2f_full_row.append(other_dep_ids[i])
    #     r2f_full_col.append(global2ind[other_dep_ids[i]])
    #     r2f_full_data.append(1)

    r2f_full_expanded = scipy.sparse.coo_array((r2f_full_data, (r2f_full_row, r2f_full_col)), shape=(
        v.shape[0]*3, all_ind_ids.shape[0]))

    print("here5")

    with h5py.File(workspace_path + "CT_bezier_r2f_with_dirichlet_expanded.hdf5", "w") as f:
        f.create_dataset("A_triplets/values", data=r2f_full_expanded.data)
        f.create_dataset("A_triplets/cols", data=r2f_full_expanded.col)
        f.create_dataset("A_triplets/rows", data=r2f_full_expanded.row)
        f.create_dataset("A_triplets/shape", data=r2f_full_expanded.shape)
        # r2f triplets

    # compute proj
    v_expanded = np.reshape(v, (v.shape[0]*3, 1)).T[0]
    v_expanded_reduce = v_expanded[all_ind_ids]
    print(v_expanded_reduce.shape)
    print(r2f_full_expanded.shape)

    # use arbi initial guess
    # proj = r2f_full_expanded @ v_expanded_reduce - v_expanded

    # use interp mesh
    interp_mesh = mio.read(initial_interp_mesh)
    interp_v = interp_mesh.points
    interp_v_expanded = np.reshape(interp_v, (interp_v.shape[0]*3, 1)).T[0]
    proj = interp_v_expanded - v_expanded

    for id in other_dep_ids:
        proj[id] = 0
    for id in other_ind_ids:
        proj[id] = 0

    # test
    test_vec = np.random.rand(v_expanded_reduce.shape[0])

    full = (r2f_full_expanded @ test_vec + proj)

    error = stacked @ full - b
    print("error: ", np.linalg.norm(error))

    with h5py.File(workspace_path + "CT_bezier_all_matrices.hdf5", "w") as f:
        f.create_dataset("A_triplets/values", data=stacked.data)
        f.create_dataset("A_triplets/cols", data=stacked.col.astype(np.int32))
        f.create_dataset("A_triplets/rows", data=stacked.row.astype(np.int32))
        f.create_dataset("A_triplets/shape", data=stacked.shape)

        f.create_dataset("A_proj_triplets/values", data=r2f_full_expanded.data)
        f.create_dataset("A_proj_triplets/cols",
                         data=r2f_full_expanded.col.astype(np.int32))
        f.create_dataset("A_proj_triplets/rows",
                         data=r2f_full_expanded.row.astype(np.int32))
        f.create_dataset("A_proj_triplets/shape", data=r2f_full_expanded.shape)

        f.create_dataset("b_proj", data=proj[:, None])

        f.create_dataset("b", data=b[:, None])
