import c1utils

def face_split(tet_file, surface_file, s2t_vmap_file, surface_adj_tet_file):
    tet_after_para_mesh = mio.read(tet_file)
    tet_after_para_vertices = tet_after_para_mesh.points.tolist()
    # tet_after_para_tets = tet_after_para_mesh.cells_dict['tetra'].tolist()

    tet_after_para_tets = tet_after_para_mesh.cells_dict["tetra"]
    tet_after_para_tets = tet_after_para_tets[:, [1, 0, 2, 3]]
    tet_after_para_tets = tet_after_para_tets.tolist()

    para_out_v, para_out_tc, _, para_out_f, para_out_ftc, _ = igl.read_obj(
        surface_file
    )

    para_out_v_to_tet_v_map = np.loadtxt(
        s2t_vmap_file
    ).astype(np.int64)

    print("para_out_v_to_tet_v_map.shape", para_out_v_to_tet_v_map.shape)

    surface_adj_tet_para_out = {}
    tet_surface_para_out = {}
    surface_adj_tet_from_file = np.loadtxt(
        surface_adj_tet_file
    ).astype(np.int64)
    print("surface_adj_tet_from_file.shape", surface_adj_tet_from_file.shape)
    for i in range(surface_adj_tet_from_file.shape[0]):
        surface_adj_tet_para_out[surface_adj_tet_from_file[i][0]] = [
            surface_adj_tet_from_file[i][1],
            surface_adj_tet_from_file[i][2],
        ]
        tet_surface_para_out[surface_adj_tet_from_file[i][1]] = [
            surface_adj_tet_from_file[i][0]
        ]
        tet_surface_para_out[surface_adj_tet_from_file[i][2]] = [
            surface_adj_tet_from_file[i][0]
        ]

    winding_numbers = tet_after_para_mesh.cell_data["winding_number"][0]

    print("[{}] ".format(datetime.datetime.now()), "Face splitting ...")
    # face split
    tet_after_face_split_tets = []
    tet_after_face_split_vertices = copy.deepcopy(tet_after_para_vertices)

    # fid contains vid
    face_split_f_to_tet_v_map = {}

    surface_tet_cnt = 0
    visited_list = []

    # new winding numbers
    new_winding_numbers = {}

    for tid in range(len(tet_after_para_tets)):
        # non surface case
        if tid not in tet_surface_para_out:
            # propagate winding number
            new_winding_numbers[len(tet_after_face_split_tets)] = winding_numbers[tid]
            tet_after_face_split_tets.append(tet_after_para_tets[tid])
            continue

        surface_tet_cnt += 1
        visited_list.append(tid)
        # print(surface_tet_cnt)

        # surface case
        t_sf = tet_surface_para_out[tid][0]
        # print(t_sf)
        f_vs = para_out_f[t_sf]
        f_vs_tet_base = [para_out_v_to_tet_v_map[vid] for vid in f_vs]
        f_vs_coords = [np.array(para_out_v[vid]) for vid in f_vs]

        # add new vertex
        new_vid = -1
        if t_sf in face_split_f_to_tet_v_map:
            new_vid = face_split_f_to_tet_v_map[t_sf]
        else:
            new_vid = len(tet_after_face_split_vertices)
            new_v_coords = (f_vs_coords[0] + f_vs_coords[1] + f_vs_coords[2]) / 3.0
            tet_after_face_split_vertices.append(new_v_coords.tolist())
            face_split_f_to_tet_v_map[t_sf] = new_vid
        assert new_vid != -1

        # add new tets
        old_tet = tet_after_para_tets[tid]
        local_ids = [-1, -1, -1, -1]
        for i in range(3):
            for j in range(4):
                if f_vs_tet_base[i] == old_tet[j]:
                    local_ids[j] = i
                    break
        assert local_ids.count(-1) == 1
        new_tets = []
        for i in range(4):
            if local_ids[i] == -1:
                continue
            new_t = copy.deepcopy(old_tet)
            new_t[i] = new_vid
            new_tets.append(new_t)
        assert len(new_tets) == 3

        for new_t in new_tets:
            # propagate winding number
            new_winding_numbers[len(tet_after_face_split_tets)] = winding_numbers[tid]
            tet_after_face_split_tets.append(new_t)

    print("[{}] ".format(datetime.datetime.now()), "Done Face Split.")
    # save tetmesh to msh, use gmsh to create high order nodes
    tet_points_after_face_split = np.array(tet_after_face_split_vertices)
    tet_cells_after_face_split = [("tetra", np.array(tet_after_face_split_tets))]
    tetmesh_after_face_split = mio.Mesh(
        tet_points_after_face_split, tet_cells_after_face_split
    )
    tetmesh_after_face_split.write(
        workspace_path + "tetmesh_after_face_split.msh", file_format="gmsh"
    )

    return new_winding_numbers

