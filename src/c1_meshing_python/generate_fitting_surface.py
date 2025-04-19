import c1utils

def upsample_points(intepolated_mesh_file, cone_area_vid_file, cone_area_fid_file, bilap_k_ring_neighbor, bilap_sample_factor):
    print("[{}] ".format(datetime.datetime.now()), "smoothing cone area")

    c_area_vertices = np.loadtxt(cone_area_vid_file).astype(np.int32)
    c_area_vertices = np.unique(c_area_vertices)

    ryan_mesh = mio.read(intepolated_mesh_file)
    cone_area_face = np.loadtxt(cone_area_fid_file).astype(np.int32)
    cone_area_vertices = []
    tris = ryan_mesh.cells[0].data
    v_upsample_local, f_upsample_local = sample1(bilap_sample_factor)
    h_nodes = []
    freeze_nodes = []
    v_upsample = []
    f_upsample = []
    offset = 0
    for i, tt in enumerate(tris):
        h_nodes.append(range(offset, offset+10))
        freeze_nodes.extend(range(offset, offset+3))

        if any(tt[k] in c_area_vertices for k in range(10)):
            # print(i)
            cone_area_vertices.extend(range(offset, offset + v_upsample_local.shape[0]))

        nodes = ryan_mesh.points[tt]
        newv = eval_lagr(v_upsample_local, nodes)
        v_upsample.append(newv)

        f_upsample.append(f_upsample_local + offset)
        offset += newv.shape[0]

    v_upsample = np.row_stack(v_upsample)
    f_upsample = np.row_stack(f_upsample)
    cone_area_vertices = np.array(cone_area_vertices)

    SV,SVI,SVJ,SF = igl.remove_duplicate_vertices(v_upsample, f_upsample, 1e-10)
    cone_area_vertices = SVJ[cone_area_vertices]
    print(cone_area_vertices.shape)
    # SF = SVJ(f_upsample)

    freeze_nodes = [SVJ[i] for i in freeze_nodes]

    h_nodes = [SVJ[i] for i in h_nodes]

    igl.write_obj("sampled_ct.obj", SV, SF)

    v_ct = SV
    f_ct = SF

    # # hack one ring
    vv_adj = igl.adjacency_list(f_ct)
    for i in range(bilap_k_ring_neighbor):
        cone_area_vertices_expanded = cone_area_vertices.tolist()
        for vv in cone_area_vertices:
            cone_area_vertices_expanded.extend(vv_adj[vv])
        cone_area_vertices_expanded = np.unique(np.array(cone_area_vertices_expanded))
        cone_area_vertices = cone_area_vertices_expanded



    # v_ct, _, _, f_ct, _, _ = igl.read_obj("CT_bilaplacian_nodes_values.obj")
    # cone_area_vertices = np.loadtxt("CT_bilaplacian_nodes_values_cone_area_vertices.txt").astype(np.int32)
    # cone_area_vertices = np.unique(cone_area_vertices)

    # # hack one ring
    # vv_adj = igl.adjacency_list(f_ct)
    # for i in range(3):
    #     cone_area_vertices_expanded = cone_area_vertices.tolist()
    #     for vv in cone_area_vertices:
    #         cone_area_vertices_expanded.extend(vv_adj[vv])
    #     cone_area_vertices_expanded = np.unique(np.array(cone_area_vertices_expanded))
    #     cone_area_vertices = cone_area_vertices_expanded
    

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
    
    # freeze input vertex
    # xyz = v_ct[freeze_nodes]
    # with open("frozen_vertices.xyz", "w") as f:
    #     for i in range(xyz.shape[0]):
    #         f.write("{} {} {}\n".format(xyz[i][0], xyz[i][1], xyz[i][2]))

    # known = np.array(freeze_nodes)
    # print(known)
    # # exit()
    # Y = v_ct[known]
    # unknown = np.setdiff1d(np.arange(0, v_ct.shape[0]), known)
    
    Aeq = scipy.sparse.csr_matrix(np.zeros((0,0)))
    Beq = np.zeros((0,3))

    print("solving bilaplacian on upsampled")
    v_smoothed = igl.min_quad_with_fixed(A, B, known, Y, Aeq, Beq, True)
    # v_smoothed = scipy.sparse.linalg.spsolve(scipy.sparse.identity(L_w.shape[0]) - 0.1 * M_inv @ L_w, v_ct)

    igl.write_obj("sampled_after_smoothing.obj", v_smoothed[1], f_ct)

    m_xx_points = ryan_mesh.points.copy()
    m_xx_cells = ryan_mesh.cells[0].data

    for i, tt in enumerate(tris):
        hn = h_nodes[i]
        for j in range(10):
            m_xx_points[tt[j]] = v_smoothed[1][hn[j]]

    m_xx = mio.Mesh(m_xx_points, [('triangle10', m_xx_cells)])
    m_xx.write("this_is_correct.msh", file_format='gmsh')

    smoothed_normals = igl.per_vertex_normals(m_xx_points, ryan_mesh.cells[0].data[:, :3], 1)
    igl.write_obj("CT_smoothed_cone.obj", m_xx_points, ryan_mesh.cells[0].data[:, :3])

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

    # debug code
    # for i, tt in enumerate(tris):
    #     A_debug = np.array([
    #             lagr0(v_upsample_local[:,0], v_upsample_local[:,1]),
    #             lagr1(v_upsample_local[:,0], v_upsample_local[:,1]),
    #             lagr2(v_upsample_local[:,0], v_upsample_local[:,1]),
    #             lagr3(v_upsample_local[:,0], v_upsample_local[:,1]),
    #             lagr4(v_upsample_local[:,0], v_upsample_local[:,1]),
    #             lagr5(v_upsample_local[:,0], v_upsample_local[:,1]),
    #             lagr6(v_upsample_local[:,0], v_upsample_local[:,1]),
    #             lagr7(v_upsample_local[:,0], v_upsample_local[:,1]),
    #             lagr8(v_upsample_local[:,0], v_upsample_local[:,1]),
    #             lagr9(v_upsample_local[:,0], v_upsample_local[:,1])
    #         ]).T
        
    #     b_debug = v_smoothed_in_patch[i*v_upsample_local.shape[0]:(i+1) *v_upsample_local.shape[0], :]

    #     ATA = A_debug.T @ A_debug
    #     ATb = A_debug.T @ b_debug

    #     node_pos_debug = igl.min_quad_with_fixed(scipy.sparse.csr_matrix(ATA), -ATb, np.arange(0,9), ryan_mesh.points[tt[np.arange(0,9)]], Aeq, Beq, True)

    #     # print(node_pos_debug)
    #     print(node_pos_debug[1] - ryan_mesh.points[tt])
    #     # break

    # exit()

    print("v_sample_local size: ",v_upsample_local.shape)
    for i, tt in enumerate(tris):
        v_s_local = v_smoothed_in_patch[i * v_upsample_local.shape[0]: (i+1) * v_upsample_local.shape[0], :]
        A_fit_local = np.array([
            lagr0(v_upsample_local[:,0], v_upsample_local[:,1]),
            lagr1(v_upsample_local[:,0], v_upsample_local[:,1]),
            lagr2(v_upsample_local[:,0], v_upsample_local[:,1]),
            lagr3(v_upsample_local[:,0], v_upsample_local[:,1]),
            lagr4(v_upsample_local[:,0], v_upsample_local[:,1]),
            lagr5(v_upsample_local[:,0], v_upsample_local[:,1]),
            lagr6(v_upsample_local[:,0], v_upsample_local[:,1]),
            lagr7(v_upsample_local[:,0], v_upsample_local[:,1]),
            lagr8(v_upsample_local[:,0], v_upsample_local[:,1]),
            lagr9(v_upsample_local[:,0], v_upsample_local[:,1])
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

    print(A_fit_rows.shape)
    print(A_fit_cols.shape)
    print(A_fit_values.shape)

    b_fit = np.array(b_fit)
    A_fit = scipy.sparse.coo_array((A_fit_values, (A_fit_rows, A_fit_cols)), shape=(b_fit.shape[0], 10*tris.shape[0]))
    # A_fit = scipy.sparse.coo_array((A_fit_values, (A_fit_rows, A_fit_cols)), shape=(b_fit.shape[0], ryan_mesh.points.shape[0]))

    P_fit = scipy.sparse.coo_array((P_fit_values, (P_fit_rows, P_fit_cols)), shape=(10 * tris.shape[0], ryan_mesh.points.shape[0]))

    A_lsq = A_fit @ P_fit
    # A_lsq = A_fit
    A_lsq = A_lsq.tocsr()


    # fitting normal
    linear_mesh = mio.read("CT_bilaplacian_nodes.obj")
    linear_points = linear_mesh.points
    A_lsq_sti = A_lsq[SVI, :]
    sti_point = A_lsq_sti @ ryan_mesh.points
    # sti_point = A_lsq_sti @ linear_points

    igl.write_obj("test_A_lsq.obj", sti_point, f_ct)

    # L_w_sti = igl.cotmatrix(v_smoothed[1], f_ct)
    # M_sti = igl.massmatrix(v_smoothed[1], f_ct)
    L_w_sti = igl.cotmatrix(A_lsq_sti @ linear_points, f_ct)
    M_sti = igl.massmatrix(A_lsq_sti @ linear_points, f_ct)

    M_inv_rows_sti = np.array([i for i in range(M_sti.shape[0])])
    M_inv_cols_sti = np.array([i for i in range(M_sti.shape[1])])
    M_inv_data_sti = np.array([(1.0 / M_sti[i, i]) for i in M_inv_rows_sti])
    M_inv_data_sti_sqrt = np.array([1.0 / np.sqrt(M_sti[i, i]) for i in M_inv_rows_sti])
    M_size_sti = len(M_inv_cols_sti)
    M_inv_sti = scipy.sparse.csc_matrix(
        (M_inv_data_sti, (M_inv_rows_sti, M_inv_cols_sti)), shape=(M_size_sti, M_size_sti)
    )
    M_inv_sti_sqrt = scipy.sparse.csc_matrix(
        (M_inv_data_sti_sqrt, (M_inv_rows_sti, M_inv_cols_sti)), shape=(M_size_sti, M_size_sti)
    )

    A_sti = M_inv_sti_sqrt @ L_w_sti @ A_lsq_sti
    # b_sti = M_inv_sti @ L_w_sti @ v_smoothed[1]
    b_sti = M_inv_sti @ L_w_sti @ (v_smoothed[1] - A_lsq_sti @ linear_points)
    # M_inv_sti @ L_w_sti @ v_ct - A_sti @ linear_points

    A_sti =  L_w_sti @ A_lsq_sti
    b_sti =  L_w_sti @ (v_smoothed[1] - A_lsq_sti @ linear_points)

    print(A_lsq_sti.shape)
    print(v_smoothed[1].shape)

    eps = 1
    A_sti_2 = M_inv_sti_sqrt @ A_lsq_sti
    b_sti_2 = M_inv_sti @ (v_smoothed[1] - A_lsq_sti @ linear_points)

    
    # exit()

    free_node_ids = c_area_vertices
    all_node_ids = np.arange(0, ryan_mesh.points.shape[0])
    fixed_node_ids = np.setdiff1d(all_node_ids, free_node_ids, True)

    A_fixed_rows = np.arange(fixed_node_ids.shape[0])
    A_fixed_cols = fixed_node_ids
    A_fixed_values = np.ones(fixed_node_ids.shape[0])

    print(A_fixed_rows.shape)
    print(A_fixed_cols.shape)
    print(A_fixed_values.shape)

    A_fixed = scipy.sparse.coo_array((A_fixed_values, (A_fixed_rows, A_fixed_cols)), shape=(fixed_node_ids.shape[0], ryan_mesh.points.shape[0]))
    b_fixed = ryan_mesh.points[fixed_node_ids]

    print(A_lsq.shape)
    # solve ATAx = ATb
    ATA = A_lsq.T @ A_lsq
    ATb = A_lsq.T @ b_fit

    print(ATA.shape)

    print("xxxxx: ",np.linalg.norm(A_lsq @ ryan_mesh.points - v_upsample))


    # print(A_lsq[28:28*2,:] @ ryan_mesh.points)
    # print(v_upsample[:28])
    i = 4
    print("xxxx: ",A_lsq[[28*i+1],:])
    print("xxx: ",tris[i])
    print(np.linalg.norm(A_lsq[28 * i:28*(i+1),:] @ ryan_mesh.points- v_upsample[28*i:28*(i+1)],axis=1))
    print(v_upsample[28*i+1])
    print(ryan_mesh.points[3])

    # lhs = scipy.sparse.vstack((ATA, A_fixed))
    # rhs = scipy.sparse.vstack((ATb, b_fixed))

    # print(lhs.shape)
    # print(rhs.shape)
    # print(lhs.count_nonzero())

    # node_pos = scipy.sparse.linalg.spsolve(lhs.tocsr(), rhs)

    rrr_x = scipy.sparse.linalg.lsqr(A_lsq, b_fit[:,0])
    rrr_y = scipy.sparse.linalg.lsqr(A_lsq, b_fit[:,1])
    rrr_z = scipy.sparse.linalg.lsqr(A_lsq, b_fit[:,2])

    rrr = np.vstack((rrr_x[0], rrr_y[0], rrr_z[0])).T
    print(rrr.shape)


    # exit()

    # node_pos = igl.min_quad_with_fixed(ATA, -ATb, fixed_node_ids, ryan_mesh.points[fixed_node_ids], Aeq, Beq, True)

    # fit_points = node_pos[1]
    fit_points = rrr
    fit_cells = ryan_mesh.cells[0].data

    m_fit = mio.Mesh(fit_points, [('triangle10', fit_cells)])
    m_fit.write("fit_p3.msh", file_format='gmsh')


    # TODO: resurrect this

    # ####################################################
    # #             Call CT with new normals             #
    # ####################################################

    # print("[{}] ".format(datetime.datetime.now()), "Calling Clough Tocher code with new normals")
    # ct_command_2 = (
    #     path_to_ct_exe
    #     + " --input "
    #     + workspace_path
    #     + "surface_uv_after_cone_split.obj -o CT --vertex_normals CT_smoothed_normals.txt"
    # )

    # subprocess.run(ct_command_2, shell=True, check=True)



def generate_soft_constraint(soft_file_name):
    # soft constraint
    print(
        "[{}] ".format(datetime.datetime.now()),
        "constructing soft constraint matrix ...",
    )

    # A_1 = L    b_1 = -L x_0
    # A_2 = I    b_2 = x0 - xtrg (for now xtrg can be zero to run some experiments)

    lap_conn_file = "CT_bilaplacian_nodes.obj"
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
    # A_2 = sparse.identity(len(local2global))
    # A_2 = M_inv.copy()
    # b_2 = np.zeros_like(v[local2global, :])  # TODO: add xtrg

    tet_cone_ids = local2global[cone_vids]

    # dis_file = np.loadtxt(output_name+"_displacements.txt")
    # A_2_rows = []
    # A_2_cols = []
    # A_2_values = []
    # for i in range(dis_file.shape[0]):
    #     if int(dis_file[i][0]) in tet_cone_ids:
    #         continue
    #     A_2_rows.append(i)
    #     A_2_cols.append(int(dis_file[i][0]))
    #     A_2_values.append(1.0)

    # A_2 = scipy.sparse.coo_array((A_2_values, (A_2_rows, A_2_cols)), shape=(dis_file.shape[0], v.shape[0]))
    # b_2 = dis_file[:,1:]

    # A_1_p = A_1.tocoo(True)
    # A_2_p = A_2.tocoo(True)

    # smoothed_mesh = mio.read("CT_smoothed_cone.obj")
    # smoothed_points =  smoothed_mesh.points
    smoothed_mesh = mio.read("fit_p3.msh")
    smoothed_points =  smoothed_mesh.points
    # smoothed_mesh = mio.read("CT_from_lagrange_nodes.msh")
    # smoothed_points = smoothed_mesh.points
    linear_mesh = mio.read("CT_bilaplacian_nodes.obj")
    linear_points = linear_mesh.points

    # b_2 = smoothed_points - linear_points
    # A_2 = scipy.sparse.identity(b_2.shape[0]).tocoo(True)

    # fit normal
    A_2 = A_sti.tocoo(True)
    b_2 = b_sti
    # print(A_2.shape)
    # print(b_2.shape)
    # exit()

    # with h5py.File("soft_1.hdf5", "w") as file:
    #     file.create_dataset("b", data=b_1)
    #     file.create_dataset("A_triplets/values", data=A_1_p.data)
    #     file.create_dataset("A_triplets/cols", data=A_1_p.col.astype(np.int32))
    #     file.create_dataset("A_triplets/rows", data=A_1_p.row.astype(np.int32))
    #     file.create_dataset("local2global", data=local2global.astype(np.int32))

    # deprecated block
    with h5py.File("soft_2.hdf5", "w") as file:
        file.create_dataset("b", data=b_2)
        file.create_dataset("A_triplets/values", data=A_2.data)
        file.create_dataset("A_triplets/cols", data=A_2.col.astype(np.int32))
        file.create_dataset("A_triplets/rows", data=A_2.row.astype(np.int32))
        file.create_dataset("local2global", data=local2global.astype(np.int32))

    # with h5py.File("soft_2.hdf5", "w") as file:
    #     file.create_dataset("b", data=b_2)
    #     file.create_dataset("A_triplets/values", data=A_2_p.data)
    #     file.create_dataset("A_triplets/cols", data=A_2_p.col.astype(np.int32))
    #     file.create_dataset("A_triplets/rows", data=A_2_p.row.astype(np.int32))

    A_3 = A_sti_2.tocoo(True)
    b_3 = b_sti_2

    with h5py.File("soft_3.hdf5", "w") as file:
        file.create_dataset("b", data=b_3)
        file.create_dataset("A_triplets/values", data=A_3.data)
        file.create_dataset("A_triplets/cols", data=A_3.col.astype(np.int32))
        file.create_dataset("A_triplets/rows", data=A_3.row.astype(np.int32))
        file.create_dataset("local2global", data=local2global.astype(np.int32))



