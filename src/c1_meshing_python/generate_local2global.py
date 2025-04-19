import c1utils

def generate_local2global():
    ct_interpolants_file = "CT_from_lagrange_nodes.msh"
    ct_input_v_to_output_v_file = "CT_from_lagrange_nodes_input_v_to_output_v_map.txt"

    print(
        "[{}] ".format(datetime.datetime.now()),
        "Doing high order tri to tet mapping ...",
    )
    # assume converted to third order in gmsh
    tetmesh_high_order = mio.read(
        workspace_path + "tetmesh_after_face_split_high_order_tet.msh"
    )
    # assume ct interpolant
    surface_high_order = mio.read(workspace_path + ct_interpolants_file)

    tetmesh_high_order_conn = tetmesh_high_order.cells[0].data
    surface_high_order_conn = surface_high_order.cells[0].data

    surface_high_order_vertices = surface_high_order.points

    # input surface to output surface v mapping
    # surface_mapping_file = "input_v_to_output_v_map.txt"
    surface_input_to_output_v_map = {}
    surface_output_to_input_v_map = {}
    with open(workspace_path + ct_input_v_to_output_v_file, "r") as file:
        for line in file:
            values = line.split()
            surface_input_to_output_v_map[int(values[0])] = int(values[1])
            surface_output_to_input_v_map[int(values[1])] = int(values[0])

    surface_high_order_conn_with_input_v_idx = copy.deepcopy(surface_high_order_conn)
    surface_high_order_vertices_with_input_v_idx = copy.deepcopy(
        surface_high_order_vertices
    )

    for i in range(surface_high_order_conn_with_input_v_idx.shape[0]):
        for j in range(2):
            # print(surface_high_order_conn_with_input_v_idx[i][j])
            surface_high_order_conn_with_input_v_idx[i][j] = (
                surface_output_to_input_v_map[
                    surface_high_order_conn_with_input_v_idx[i][j]
                ]
            )

    for i in range(surface_high_order_vertices_with_input_v_idx.shape[0]):
        if i in surface_output_to_input_v_map:
            surface_high_order_vertices_with_input_v_idx[
                surface_output_to_input_v_map[i]
            ] = surface_high_order_vertices[i]

    # map the barycenter vertex of each input tri to tet
    assert surface_high_order_conn_with_input_v_idx.shape[0] % 3 == 0
    bc_surface_to_tet_map = {}
    bc_tet_to_surface_map = {}

    for i in range(surface_high_order_conn_with_input_v_idx.shape[0] // 3):
        tet_split_vid = face_split_f_to_tet_v_map[i]
        bc_surface_to_tet_map[
            surface_high_order_conn_with_input_v_idx[i * 3 + 0][2]
        ] = tet_split_vid
        bc_tet_to_surface_map[tet_split_vid] = surface_high_order_conn_with_input_v_idx[
            i * 3 + 0
        ][2]

    print(
        "[{}] ".format(datetime.datetime.now()),
        "breaking done high order tet conn into face/edge to vertex mappings",
    )
    # break down high order tets into face and edge representation
    tet_edge_to_vertices = {}
    tet_face_to_vertices = {}

    # 0 1 2 3 v
    # 4 5 e01
    # 6 7 e12
    # 8 9 e20
    # 10 11 e30
    # 12 13 e32
    # 14 15 e31
    # 16 f012
    # 17 f013
    # 18 f023
    # 19 f123

    for tet in tetmesh_high_order_conn:
        e01 = str(tet[0]) + "+" + str(tet[1])
        e10 = str(tet[1]) + "+" + str(tet[0])
        if e01 in tet_edge_to_vertices:
            assert tet_edge_to_vertices[e01] == [tet[4], tet[5]]
        else:
            tet_edge_to_vertices[e01] = [tet[4], tet[5]]
            tet_edge_to_vertices[e10] = [tet[5], tet[4]]

        e12 = str(tet[1]) + "+" + str(tet[2])
        e21 = str(tet[2]) + "+" + str(tet[1])
        if e12 in tet_edge_to_vertices:
            assert tet_edge_to_vertices[e12] == [tet[6], tet[7]]
        else:
            tet_edge_to_vertices[e12] = [tet[6], tet[7]]
            tet_edge_to_vertices[e21] = [tet[7], tet[6]]

        e20 = str(tet[2]) + "+" + str(tet[0])
        e02 = str(tet[0]) + "+" + str(tet[2])
        if e20 in tet_edge_to_vertices:
            assert tet_edge_to_vertices[e20] == [tet[8], tet[9]]
        else:
            tet_edge_to_vertices[e20] = [tet[8], tet[9]]
            tet_edge_to_vertices[e02] = [tet[9], tet[8]]

        e30 = str(tet[3]) + "+" + str(tet[0])
        e03 = str(tet[0]) + "+" + str(tet[3])
        if e30 in tet_edge_to_vertices:
            assert tet_edge_to_vertices[e30] == [tet[10], tet[11]]
        else:
            tet_edge_to_vertices[e30] = [tet[10], tet[11]]
            tet_edge_to_vertices[e03] = [tet[11], tet[10]]

        e32 = str(tet[3]) + "+" + str(tet[2])
        e23 = str(tet[2]) + "+" + str(tet[3])
        if e32 in tet_edge_to_vertices:
            assert tet_edge_to_vertices[e32] == [tet[12], tet[13]]
        else:
            tet_edge_to_vertices[e32] = [tet[12], tet[13]]
            tet_edge_to_vertices[e23] = [tet[13], tet[12]]

        e31 = str(tet[3]) + "+" + str(tet[1])
        e13 = str(tet[1]) + "+" + str(tet[3])
        if e31 in tet_edge_to_vertices:
            assert tet_edge_to_vertices[e31] == [tet[14], tet[15]]
        else:
            tet_edge_to_vertices[e31] = [tet[14], tet[15]]
            tet_edge_to_vertices[e13] = [tet[15], tet[14]]

        f012 = [tet[0], tet[1], tet[2]]
        f013 = [tet[0], tet[1], tet[3]]
        f023 = [tet[0], tet[2], tet[3]]
        f123 = [tet[1], tet[2], tet[3]]
        f012.sort()
        f013.sort()
        f023.sort()
        f123.sort()
        f012_str = str(f012[0]) + "+" + str(f012[1]) + "+" + str(f012[2])
        f013_str = str(f013[0]) + "+" + str(f013[1]) + "+" + str(f013[2])
        f023_str = str(f023[0]) + "+" + str(f023[1]) + "+" + str(f023[2])
        f123_str = str(f123[0]) + "+" + str(f123[1]) + "+" + str(f123[2])

        tet_face_to_vertices[f012_str] = tet[16]
        tet_face_to_vertices[f013_str] = tet[17]
        tet_face_to_vertices[f023_str] = tet[18]
        tet_face_to_vertices[f123_str] = tet[19]

    print(
        "[{}] ".format(datetime.datetime.now()), "constructing tri <-> tet v mappings"
    )
    # map high order tri vertices to tet vertices
    tri_to_tet_high_order_v_map = {}
    tet_to_tri_high_order_v_map = {}

    for tri in surface_high_order_conn_with_input_v_idx:
        # print(tri)
        vs = [
            para_out_v_to_tet_v_map[tri[0]],
            para_out_v_to_tet_v_map[tri[1]],
            bc_surface_to_tet_map[tri[2]],
        ]  # vertices in tet idx
        face = [vs[0], vs[1], vs[2]]
        face.sort()
        face = str(face[0]) + "+" + str(face[1]) + "+" + str(face[2])

        e01 = str(vs[0]) + "+" + str(vs[1])
        e12 = str(vs[1]) + "+" + str(vs[2])
        e20 = str(vs[2]) + "+" + str(vs[0])

        # face
        tri_to_tet_high_order_v_map[tri[9]] = tet_face_to_vertices[face]

        # edges
        tri_to_tet_high_order_v_map[tri[3]] = tet_edge_to_vertices[e01][0]
        tri_to_tet_high_order_v_map[tri[4]] = tet_edge_to_vertices[e01][1]
        tri_to_tet_high_order_v_map[tri[5]] = tet_edge_to_vertices[e12][0]
        tri_to_tet_high_order_v_map[tri[6]] = tet_edge_to_vertices[e12][1]
        tri_to_tet_high_order_v_map[tri[7]] = tet_edge_to_vertices[e20][0]
        tri_to_tet_high_order_v_map[tri[8]] = tet_edge_to_vertices[e20][1]

        # vertices
        tri_to_tet_high_order_v_map[tri[0]] = para_out_v_to_tet_v_map[tri[0]]
        tri_to_tet_high_order_v_map[tri[1]] = para_out_v_to_tet_v_map[tri[1]]
        tri_to_tet_high_order_v_map[tri[2]] = bc_surface_to_tet_map[tri[2]]

    for key in tri_to_tet_high_order_v_map:
        tet_to_tri_high_order_v_map[tri_to_tet_high_order_v_map[key]] = key

    # assign high order vertex coordinates to high order tetmesh
    curved_tet_conn = copy.deepcopy(tetmesh_high_order_conn)
    curved_tet_vertices = copy.deepcopy(tetmesh_high_order.points)

    for key in tet_to_tri_high_order_v_map:
        curved_tet_vertices[key] = surface_high_order_vertices_with_input_v_idx[
            tet_to_tri_high_order_v_map[key]
        ]

    # write files
    print("[{}] ".format(datetime.datetime.now()), "writing outputs ...")
    # TODO: file names
    curved_tet_file_name = output_name + "_curved_tetmesh.msh"
    linear_tet_file_name = output_name + "_initial_tetmesh.msh"
    displacement_file_name = output_name + "_displacements.txt"
    tri_to_tet_index_mapping_file_name = output_name + "_tri_to_tet_v_map.txt"

    curved_points = curved_tet_vertices
    curved_cells = [("tetra20", curved_tet_conn)]

    curved_tetmesh = mio.Mesh(curved_points, curved_cells)
    curved_tetmesh.write(workspace_path + curved_tet_file_name, file_format="gmsh")

    linear_tetmesh = mio.Mesh(tetmesh_high_order.points, tetmesh_high_order.cells)
    linear_tetmesh.write(workspace_path + linear_tet_file_name, file_format="gmsh")

    delta_mesh_points = curved_tetmesh.points - linear_tetmesh.points
    delta_mesh_cells = []
    delta_mesh = mio.Mesh(delta_mesh_points, curved_cells)

    with open(workspace_path + displacement_file_name, "w") as file:
        for key in tet_to_tri_high_order_v_map:
            file.write(
                f"{key} {delta_mesh_points[key][0]} {delta_mesh_points[key][1]} {delta_mesh_points[key][2]}\n"
            )

    # write the tri to tet vertex map
    tri_to_tet_idx_file_correct = open(
        workspace_path + tri_to_tet_index_mapping_file_name, "w"
    )

    tri_to_tet_constraint_nodes_v_map = {}

    for i in range(len(tri_to_tet_high_order_v_map.keys())):
        if i in surface_input_to_output_v_map.keys():
            tri_to_tet_constraint_nodes_v_map[surface_input_to_output_v_map[i]] = (
                tri_to_tet_high_order_v_map[i]
            )
        else:
            tri_to_tet_constraint_nodes_v_map[i] = tri_to_tet_high_order_v_map[i]

    for i in range(len(tri_to_tet_high_order_v_map.keys())):
        tri_to_tet_idx_file_correct.write(
            str(tri_to_tet_constraint_nodes_v_map[i]) + "\n"
        )

    tri_to_tet_idx_file_correct.close()

    # newly added
    # write tets incident to surface vertices
    tet_to_tri_constraint_nodes_v_map = {}
    for key, value in tri_to_tet_constraint_nodes_v_map.items():
        tet_to_tri_constraint_nodes_v_map[value] = key

    debug_cells = []
    with open("surface_adjcent_tet.txt", "w") as file:
        cells = linear_tetmesh.cells_dict["tetra20"]
        for i in range(cells.shape[0]):
            incident = False
            for node in cells[i][0:4]:
                if node in tet_to_tri_constraint_nodes_v_map:
                    incident = True
                    break
            if incident:
                # file.write(str(i) + " 3\n")
                file.write("3\n")
                debug_cells.append(cells[i])
            else:
                # file.write(str(i) + " 1\n")
                file.write("1\n")

    with open("surface_adjcent_tet_with_tid.txt", "w") as file:
        cells = linear_tetmesh.cells_dict["tetra20"]
        for i in range(cells.shape[0]):
            incident = False
            for node in cells[i][0:4]:
                if node in tet_to_tri_constraint_nodes_v_map:
                    incident = True
                    break
            if incident:
                file.write(str(i) + " 3\n")
            else:
                file.write(str(i) + " 1\n")

    debug_incident_tetmesh = mio.Mesh(linear_tetmesh.points, [("tetra20", np.array(debug_cells))])
    debug_incident_tetmesh.write("debug_incident_tetmesh.msh", file_format='gmsh')
