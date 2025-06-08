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


def upsample_and_smooth_cones(cone_area_vertices_file, cone_area_faces_file, smooth_initial_mesh_file, sample_factor, k_ring_factor):
    print("[{}] ".format(datetime.datetime.now()),
          "upsample and smoothing cone area")

    c_area_vertices = np.loadtxt(
        cone_area_vertices_file).astype(np.int32)
    c_area_vertices = np.unique(c_area_vertices)

    ryan_mesh = mio.read(smooth_initial_mesh_file)
    cone_area_face = np.loadtxt(
        cone_area_faces_file).astype(np.int32)
    cone_area_vertices = []
    tris = ryan_mesh.cells[0].data
    v_upsample_local, f_upsample_local = sample1(sample_factor)

    v_upsample_local_3d = np.array([[v_upsample_local[i][0], v_upsample_local[i][1], 0]
                                    for i in range(v_upsample_local.shape[0])])
    igl.write_obj("upsample_local.obj", v_upsample_local_3d, f_upsample_local)

    h_nodes = []
    freeze_nodes = []
    v_upsample = []
    f_upsample = []
    offset = 0

    # upsample
    for i, tt in enumerate(tris):
        h_nodes.append(range(offset, offset+10))
        freeze_nodes.extend(range(offset, offset+3))

        if any(tt[k] in c_area_vertices for k in range(10)):
            # print(i)
            cone_area_vertices.extend(
                range(offset, offset + v_upsample_local.shape[0]))

        nodes = ryan_mesh.points[tt]
        newv = eval_lagr(v_upsample_local, nodes)
        v_upsample.append(newv)

        f_upsample.append(f_upsample_local + offset)
        offset += newv.shape[0]

    v_upsample = np.row_stack(v_upsample)
    f_upsample = np.row_stack(f_upsample)
    cone_area_vertices = np.array(cone_area_vertices)

    SV, SVI, SVJ, SF = igl.remove_duplicate_vertices(
        v_upsample, f_upsample, 1e-10)
    cone_area_vertices = SVJ[cone_area_vertices]
    # print(cone_area_vertices.shape)
    # SF = SVJ(f_upsample)

    freeze_nodes = [SVJ[i] for i in freeze_nodes]

    h_nodes = [SVJ[i] for i in h_nodes]

    igl.write_obj("sampled_ct.obj", SV, SF)

    v_ct = SV
    f_ct = SF

    # # hack one ring
    vv_adj = igl.adjacency_list(f_ct)
    for i in range(k_ring_factor):
        cone_area_vertices_expanded = cone_area_vertices.tolist()
        for vv in cone_area_vertices:
            cone_area_vertices_expanded.extend(vv_adj[vv])
        cone_area_vertices_expanded = np.unique(
            np.array(cone_area_vertices_expanded))
        cone_area_vertices = cone_area_vertices_expanded

    # laplacian smoothing
    L_w = igl.cotmatrix(v_ct, f_ct)
    M = igl.massmatrix(v_ct, f_ct)

    # assemble M_inv
    M_inv_rows = np.array([i for i in range(M.shape[0])])
    M_inv_cols = np.array([i for i in range(M.shape[1])])
    M_inv_data = np.array([1.0 / M[i, i] for i in M_inv_rows])
    M_size = len(M_inv_cols)
    M_inv = scipy.sparse.csc_matrix(
        (M_inv_data, (M_inv_rows, M_inv_cols)), shape=(M_size, M_size)
    )
    non_cone_area_vertices = []
    for i in range(v_ct.shape[0]):
        if i not in cone_area_vertices:
            non_cone_area_vertices.append(i)
    non_cone_area_vertices = np.array(non_cone_area_vertices)

    A = L_w @ M_inv @ L_w
    B = np.zeros((A.shape[0], 3))
    known = non_cone_area_vertices
    Y = v_ct[known]
    unknown = cone_area_vertices

    Aeq = scipy.sparse.csr_matrix(np.zeros((0, 0)))
    Beq = np.zeros((0, 3))

    print("solving bilaplacian on upsampled")
    v_smoothed = igl.min_quad_with_fixed(A, B, known, Y, Aeq, Beq, True)

    igl.write_obj("sampled_after_smoothing.obj", v_smoothed[1], f_ct)

    m_xx_points = ryan_mesh.points.copy()
    m_xx_cells = ryan_mesh.cells[0].data

    for i, tt in enumerate(tris):
        hn = h_nodes[i]
        for j in range(10):
            m_xx_points[tt[j]] = v_smoothed[1][hn[j]]

    m_xx = mio.Mesh(m_xx_points, [('triangle10', m_xx_cells)])
    m_xx.write("soft_target_surface_mesh.msh", file_format='gmsh')

    smoothed_normals = igl.per_vertex_normals(
        m_xx_points, ryan_mesh.cells[0].data[:, :3], 1)
    # igl.write_obj("CT_smoothed_cone.obj", m_xx_points,
    #               ryan_mesh.cells[0].data[:, :3])

    np.savetxt("CT_smoothed_normals.txt", smoothed_normals)

    xyz = v_ct[unknown]
    with open("unsmoothed_cones.xyz", "w") as f:
        for i in range(xyz.shape[0]):
            f.write("{} {} {}\n".format(xyz[i][0], xyz[i][1], xyz[i][2]))

    # fit poly
    print("fit control points to sampled cubic polynomial")
    v_smoothed_in_patch = np.zeros(v_upsample.shape)
    for i in range(v_smoothed_in_patch.shape[0]):
        v_smoothed_in_patch[i] = v_smoothed[1][SVJ[i]]

    assert v_smoothed_in_patch.shape[0] == v_upsample_local.shape[0] * tris.shape[0]

    A_fit_rows = []
    A_fit_cols = []
    A_fit_values = []
    P_fit_rows = []
    P_fit_cols = []
    P_fit_values = []
    b_fit = []

    # print("v_sample_local size: ", v_upsample_local.shape)
    for i, tt in enumerate(tris):
        v_s_local = v_smoothed_in_patch[i * v_upsample_local.shape[0]                                        :(i+1) * v_upsample_local.shape[0], :]
        A_fit_local = np.array([
            lagr0(v_upsample_local[:, 0], v_upsample_local[:, 1]),
            lagr1(v_upsample_local[:, 0], v_upsample_local[:, 1]),
            lagr2(v_upsample_local[:, 0], v_upsample_local[:, 1]),
            lagr3(v_upsample_local[:, 0], v_upsample_local[:, 1]),
            lagr4(v_upsample_local[:, 0], v_upsample_local[:, 1]),
            lagr5(v_upsample_local[:, 0], v_upsample_local[:, 1]),
            lagr6(v_upsample_local[:, 0], v_upsample_local[:, 1]),
            lagr7(v_upsample_local[:, 0], v_upsample_local[:, 1]),
            lagr8(v_upsample_local[:, 0], v_upsample_local[:, 1]),
            lagr9(v_upsample_local[:, 0], v_upsample_local[:, 1])
        ])
        # print(A_fit_local.shape)
        A_fit_local = A_fit_local.T
        # print(A_fit_local.shape)
        # print(v_upsample_local.shape[0])
        assert A_fit_local.shape[0] == v_upsample_local.shape[0]
        assert A_fit_local.shape[1] == 10

        nodes = ryan_mesh.points[tt]
        newv = eval_lagr(v_upsample_local, nodes)

        # if np.linalg.norm(A_fit_local@ nodes -newv) > 1e-15:
        #     print("test: ", np.linalg.norm(A_fit_local@ nodes -newv))

        for k in range(A_fit_local.shape[0]):
            for h in range(A_fit_local.shape[1]):
                # block i
                A_fit_rows.append(i * A_fit_local.shape[0] + k)
                A_fit_cols.append(i * A_fit_local.shape[1] + h)
                A_fit_values.append(A_fit_local[k][h])

        # for k in range(A_fit_local.shape[0]):
        #     for h in range(A_fit_local.shape[1]):
        #         # block i
        #         A_fit_rows.append(i * A_fit_local.shape[0] + k)
        #         A_fit_cols.append(tt[h])
        #         A_fit_values.append(A_fit_local[k][h])

        for k in range(A_fit_local.shape[1]):
            P_fit_rows.append(i * A_fit_local.shape[1] + k)
            P_fit_cols.append(tt[k])
            P_fit_values.append(1.0)

        for k in range(A_fit_local.shape[0]):
            b_fit.append(v_s_local[k])

    A_fit_rows = np.array(A_fit_rows)
    A_fit_cols = np.array(A_fit_cols)
    A_fit_values = np.array(A_fit_values)
    P_fit_rows = np.array(P_fit_rows)
    P_fit_cols = np.array(P_fit_cols)
    P_fit_values = np.array(P_fit_values)

    # print(A_fit_rows.shape)
    # print(A_fit_cols.shape)
    # print(A_fit_values.shape)

    b_fit = np.array(b_fit)
    A_fit = scipy.sparse.coo_array((A_fit_values, (A_fit_rows, A_fit_cols)), shape=(
        b_fit.shape[0], 10*tris.shape[0]))
    # A_fit = scipy.sparse.coo_array((A_fit_values, (A_fit_rows, A_fit_cols)), shape=(b_fit.shape[0], ryan_mesh.points.shape[0]))

    P_fit = scipy.sparse.coo_array((P_fit_values, (P_fit_rows, P_fit_cols)), shape=(
        10 * tris.shape[0], ryan_mesh.points.shape[0]))

    A_lsq = A_fit @ P_fit
    # A_lsq = A_fit
    A_lsq = A_lsq.tocsr()

    # fitting normal
    linear_mesh = mio.read("CT_bilaplacian_nodes.obj")
    linear_points = linear_mesh.points
    A_lsq_sti = A_lsq[SVI, :]  # sti stands for stiched
    sti_point = A_lsq_sti @ ryan_mesh.points
    # sti_point = A_lsq_sti @ linear_points

    igl.write_obj("test_A_lsq.obj", sti_point, f_ct)

    # L_w_sti = igl.cotmatrix(v_smoothed[1], f_ct)
    # M_sti = igl.massmatrix(v_smoothed[1], f_ct)

    # TODO: this is changed for test, change this back if not working. seems nothing different because mass matrix is not used
    L_w_sti = igl.cotmatrix(A_lsq_sti @ linear_points, f_ct)
    M_sti = igl.massmatrix(A_lsq_sti @ linear_points, f_ct)
    # L_w_sti = igl.cotmatrix(A_lsq_sti @ ryan_mesh.points, f_ct)
    # M_sti = igl.massmatrix(A_lsq_sti @ ryan_mesh.points, f_ct)

    M_inv_rows_sti = np.array([i for i in range(M_sti.shape[0])])
    M_inv_cols_sti = np.array([i for i in range(M_sti.shape[1])])
    M_inv_data_sti = np.array([(1.0 / M_sti[i, i]) for i in M_inv_rows_sti])
    M_inv_data_sti_sqrt = np.array(
        [1.0 / np.sqrt(M_sti[i, i]) for i in M_inv_rows_sti])
    M_size_sti = len(M_inv_cols_sti)
    M_inv_sti = scipy.sparse.csc_matrix(
        (M_inv_data_sti, (M_inv_rows_sti, M_inv_cols_sti)), shape=(
            M_size_sti, M_size_sti)
    )
    M_inv_sti_sqrt = scipy.sparse.csc_matrix(
        (M_inv_data_sti_sqrt, (M_inv_rows_sti, M_inv_cols_sti)), shape=(
            M_size_sti, M_size_sti)
    )

    A_sti = M_inv_sti_sqrt @ L_w_sti @ A_lsq_sti
    # b_sti = M_inv_sti @ L_w_sti @ v_smoothed[1]
    b_sti = M_inv_sti @ L_w_sti @ (v_smoothed[1] - A_lsq_sti @ linear_points)
    # M_inv_sti @ L_w_sti @ v_ct - A_sti @ linear_points

    # TODO: uncomment this back
    A_sti = L_w_sti @ A_lsq_sti
    b_sti = L_w_sti @ (v_smoothed[1] - A_lsq_sti @ linear_points)

    # print(A_lsq_sti.shape)
    # print(v_smoothed[1].shape)

    # eps = 1
    # A_sti_2 = M_inv_sti_sqrt @ A_lsq_sti
    # b_sti_2 = M_inv_sti @ (v_smoothed[1] - A_lsq_sti @ linear_points)

    # exit()

    free_node_ids = c_area_vertices
    all_node_ids = np.arange(0, ryan_mesh.points.shape[0])
    fixed_node_ids = np.setdiff1d(all_node_ids, free_node_ids, True)

    A_fixed_rows = np.arange(fixed_node_ids.shape[0])
    A_fixed_cols = fixed_node_ids
    A_fixed_values = np.ones(fixed_node_ids.shape[0])

    # print(A_fixed_rows.shape)
    # print(A_fixed_cols.shape)
    # print(A_fixed_values.shape)

    A_fixed = scipy.sparse.coo_array((A_fixed_values, (A_fixed_rows, A_fixed_cols)), shape=(
        fixed_node_ids.shape[0], ryan_mesh.points.shape[0]))
    b_fixed = ryan_mesh.points[fixed_node_ids]

    # print(A_lsq.shape)
    # solve ATAx = ATb
    ATA = A_lsq.T @ A_lsq
    ATb = A_lsq.T @ b_fit

    # print(ATA.shape)

    # print("xxxxx: ", np.linalg.norm(A_lsq @ ryan_mesh.points - v_upsample))

    # print(A_lsq[28:28*2,:] @ ryan_mesh.points)
    # print(v_upsample[:28])
    # i = 4
    # print("xxxx: ", A_lsq[[28*i+1], :])
    # print("xxx: ", tris[i])
    # print(np.linalg.norm(A_lsq[28 * i:28*(i+1), :] @
    #       ryan_mesh.points - v_upsample[28*i:28*(i+1)], axis=1))
    # print(v_upsample[28*i+1])
    # print(ryan_mesh.points[3])

    # lhs = scipy.sparse.vstack((ATA, A_fixed))
    # rhs = scipy.sparse.vstack((ATb, b_fixed))

    # print(lhs.shape)
    # print(rhs.shape)
    # print(lhs.count_nonzero())

    # node_pos = scipy.sparse.linalg.spsolve(lhs.tocsr(), rhs)

    rrr_x = scipy.sparse.linalg.lsqr(A_lsq, b_fit[:, 0])
    rrr_y = scipy.sparse.linalg.lsqr(A_lsq, b_fit[:, 1])
    rrr_z = scipy.sparse.linalg.lsqr(A_lsq, b_fit[:, 2])

    rrr = np.vstack((rrr_x[0], rrr_y[0], rrr_z[0])).T
    # print(rrr.shape)

    # exit()

    # node_pos = igl.min_quad_with_fixed(ATA, -ATb, fixed_node_ids, ryan_mesh.points[fixed_node_ids], Aeq, Beq, True)

    # fit_points = node_pos[1]
    fit_points = rrr
    fit_cells = ryan_mesh.cells[0].data

    m_fit = mio.Mesh(fit_points, [('triangle10', fit_cells)])
    m_fit.write("fit_p3.msh", file_format='gmsh')

    # # uniform laplacian smoothing
    # print("compute uniform laplacian")
    # adj_mat = igl.adjacency_matrix(f_ct)
    # adj_mat_sum = np.asarray(np.sum(adj_mat, axis=1)).flatten()

    # adj_diag = np.diag(adj_mat_sum)
    # adj_mat_sp = adj_mat
    # adj_diag_sp = scipy.sparse.csr_matrix(adj_diag, shape=adj_diag.shape)
    # L_uniform = adj_mat_sp - adj_diag_sp

    # A_sti_2 = L_uniform @ A_lsq_sti

    # v_sti_2 = A_lsq_sti @ linear_points

    # # L_uniform_sp = scipy.sparse.csr_matrix(L_uniform, shape=L_uniform.shape)
    # L_uniform_sp = L_uniform

    # b_sti_2 = -L_uniform_sp @ v_sti_2

    # print(A_sti_2.shape)
    # print(b_sti_2.shape)

    return A_sti, b_sti


def soft_constraint_fit_normal(workspace_path, tri_to_tet_index_mapping_file, linear_tet_file_name, A_sti, b_sti, b2l_mat_file="CT_bezier_to_lag_convertion_matrix.txt", lap_conn_file="CT_bilaplacian_nodes.obj"):
    local2global = np.loadtxt(
        workspace_path + tri_to_tet_index_mapping_file
    ).astype(np.int32)

    m = mio.read(workspace_path + linear_tet_file_name)
    v = m.points

    v_lap, _, _, f_lap, _, _ = igl.read_obj(lap_conn_file)

    L_w = igl.cotmatrix(v_lap, f_lap)
    M = igl.massmatrix(v_lap, f_lap)

    # assemble M_inv
    M_inv_rows = np.array([i for i in range(M.shape[0])])
    M_inv_cols = np.array([i for i in range(M.shape[1])])
    M_inv_data = np.array([1.0 / M[i, i] for i in M_inv_rows])
    M_size = len(M_inv_cols)
    M_inv = sparse.csc_matrix(
        (M_inv_data, (M_inv_rows, M_inv_cols)), shape=(M_size, M_size)
    )

    L = M_inv @ L_w

    A_1 = L
    b_1 = -L @ v[local2global, :]

    # smoothed_mesh = mio.read("CT_smoothed_cone.obj")
    # smoothed_points =  smoothed_mesh.points
    smoothed_mesh = mio.read("fit_p3.msh")
    smoothed_points = smoothed_mesh.points
    # smoothed_mesh = mio.read("CT_from_lagrange_nodes.msh")
    # smoothed_points = smoothed_mesh.points
    linear_mesh = mio.read("CT_bilaplacian_nodes.obj")
    linear_points = linear_mesh.points

    A_3 = A_sti.tocoo(True)
    b_3 = b_sti

    # apply bezier to lagrange convertion
    b2l_full_mat = scipy.io.mmread(b2l_mat_file)
    A_3 = (A_sti @ b2l_full_mat).tocoo(True)

    # print("soft shape: ", A_3.shape)
    # print("A_sti: ", A_sti.shape)
    # print("#v: ", v.shape[0])

    with h5py.File("soft.hdf5", "w") as file:
        file.create_dataset("b", data=b_3)
        file.create_dataset("A_triplets/values", data=A_3.data)
        file.create_dataset("A_triplets/cols", data=A_3.col.astype(np.int32))
        file.create_dataset("A_triplets/rows", data=A_3.row.astype(np.int32))
        file.create_dataset("A_triplets/shape", data=A_3.shape)

        file.create_dataset("local2global", data=local2global.astype(np.int32))

    # uniform laplacian
    # A_uni = A_sti_2.tocoo(True)
    # b_uni = b_sti_2
    # with h5py.File("soft_uni.hdf5", "w") as file:
    #     file.create_dataset("b", data=b_uni)
    #     file.create_dataset("A_triplets/values", data=A_uni.data)
    #     file.create_dataset("A_triplets/cols", data=A_uni.col.astype(np.int32))
    #     file.create_dataset("A_triplets/rows", data=A_uni.row.astype(np.int32))
    #     file.create_dataset("A_triplets/shape", data=A_uni.shape)

    #     file.create_dataset("local2global", data=local2global.astype(np.int32))


def soft_constraint_cubic_optimization(workspace_path, tri_to_tet_index_mapping_file, linear_tet_file_name, lapbel_mesh_file, lap_mesh_file, l2b_matrix_file):
    local2global = np.loadtxt(
        workspace_path + tri_to_tet_index_mapping_file
    ).astype(np.int32)

    m = mio.read(workspace_path + linear_tet_file_name)
    v = m.points

    # lap bel
    lapbel_mesh = mio.read(lapbel_mesh_file)
    lapbel_v = lapbel_mesh.points

    assert (local2global.shape[0] == lapbel_v.shape[0])

    l2b_mat = scipy.io.mmread(l2b_matrix_file)

    A = scipy.sparse.identity(lapbel_v.shape[0]).tocoo(True)
    b = l2b_mat @ lapbel_v - v[local2global]

    with h5py.File("soft.hdf5", "w") as file:
        file.create_dataset("b", data=b)
        file.create_dataset("A_triplets/values", data=A.data)
        file.create_dataset("A_triplets/cols", data=A.col.astype(np.int32))
        file.create_dataset("A_triplets/rows", data=A.row.astype(np.int32))
        file.create_dataset("A_triplets/shape", data=A.shape)

        file.create_dataset("local2global", data=local2global.astype(np.int32))

    # lap
    lap_mesh = mio.read(lap_mesh_file)
    lap_v = lap_mesh.points

    A_2 = scipy.sparse.identity(lapbel_v.shape[0]).tocoo(True)
    b_2 = l2b_mat @ lap_v - v[local2global]

    with h5py.File("soft_lap.hdf5", "w") as file:
        file.create_dataset("b", data=b_2)
        file.create_dataset("A_triplets/values", data=A_2.data)
        file.create_dataset("A_triplets/cols", data=A_2.col.astype(np.int32))
        file.create_dataset("A_triplets/rows", data=A_2.row.astype(np.int32))
        file.create_dataset("A_triplets/shape", data=A_2.shape)

        file.create_dataset("local2global", data=local2global.astype(np.int32))


def call_CT_code_with_normals(workspace_path, path_to_ct_exe, meshfile, normals_file):
    print("[{}] ".format(datetime.datetime.now()),
          "Calling Clough Tocher code with normals")
    ct_command = (
        path_to_ct_exe
        + " --input "
        + workspace_path
        + meshfile + " -o CT "
        + "--vertex_normals " + normals_file
    )

    subprocess.run(ct_command, shell=True, check=True)
