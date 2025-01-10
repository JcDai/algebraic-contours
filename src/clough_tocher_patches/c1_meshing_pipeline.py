import igl
import meshio as mio
import numpy as np
import copy 
import subprocess
import sys
import gmsh
import h5py
from scipy import sparse 
import json

# check orient3d > 0
def orient3d(aa, bb, cc, dd):
    a = np.array(aa)
    b = np.array(bb)
    c = np.array(cc)
    d = np.array(dd)
    mat = np.zeros([3, 3])
    mat[0,:] = b-a
    mat[1,:] = c-a
    mat[2,:] = d-a

    return np.linalg.det(mat) > 0

# check if face is contained by a tet
def face_in_tet(f, t):
    return all(ff in t for ff in f)

# check if two face are the same
def face_equal(f0, f1):
    ff0 = copy.deepcopy(f0.tolist())
    ff1 = copy.deepcopy(f1.tolist())
    ff0.sort()
    ff1.sort()
    if ff0 == ff1:
        return True
    return False

# check if D is in ABC
def on_tri(A, B, C, D, eps=1e-10):
    AB = B - A
    AC = C - A
    AD = D - A

    # check coplanar 
    AB_n = AB/np.linalg.norm(AB)
    AC_n = AC/np.linalg.norm(AC)
    AD_n = AD/np.linalg.norm(AD)

    r = AD_n @ (AB_n.cross(AC_n))
    if abs(r) > eps:
        return False
    
    # check in shape
    c1 = (B-A).cross(D-A)
    c2 = (C-B).cross(D-B)
    c3 = (A-C).cross(D-C)

    if c1 @ c2 > 0 and c1 @ c3 > 0:
        return True
    
    return False

if __name__ == "__main__":
    args = sys.argv

    input_file = args[1] # vtu tetmesh file with 'winding_number' as cell data
    output_name = args[2] # output name
    path_to_para_exe = args[3] # path to parametrization bin
    path_to_ct_exe = args[4] # path to Clough Tocher constraints bin
    path_to_polyfem_exe = args[5] # path to polyfem bin

    # workspace_path = args[4] # workspace path
    workspace_path = "./"

    tm = mio.read(input_file)
    vertices = tm.points
    tets = tm.cells_dict['tetra']
    filtered_tets = []
    winding_numbers = tm.cell_data['winding_number'][0]
    for i in range(tets.shape[0]):
        if abs(winding_numbers[i]) >= 0.5:
            filtered_tets.append(tets[i])

    filtered_tets = np.array(filtered_tets)

    # check orientation, TODO: only do in debug
    for i in range(tets.shape[0]):
        if not orient3d(vertices[tets[i][0]], vertices[tets[i][1]], vertices[tets[i][2]], vertices[tets[i][3]]):
            print("tet {} flipped".format(i))
        assert orient3d(vertices[tets[i][0]], vertices[tets[i][1]], vertices[tets[i][2]], vertices[tets[i][3]])
    print("passed orientation check.")

    # get surface mesh for parametrization
    surface_tet_faces = igl.boundary_facets(filtered_tets)
    para_in_v, para_in_f, im, para_in_v_to_tet_v_map = igl.remove_unreferenced(vertices, surface_tet_faces)
    assert((igl.bfs_orient(para_in_f)[0] == para_in_f).all())

    igl.write_obj(workspace_path + "embedded_surface.obj", para_in_v, para_in_f)
    print("generated embedded_surface.obj for parametrization.")

    # get tets containing surface
    print("constructing tet->surface and surface->tet mapping ...")
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
            str(ff3[0]) + "+" + str(ff3[1]) + "+" + str(ff3[2])
        ]
        for f_str in ffs:
            if f_str in surface_in_tet_map:
                surface_in_tet_map[f_str].append(i)
            else:
                surface_in_tet_map[f_str] = [i]

    print("computed face in tet map")
    # print(surface_in_tet_map)

    surface_adj_tet = {}
    tet_surface_origin = {}

    for i in range(surface_tet_faces.shape[0]):
        surface_adj_tet[i] = []

    for j in range(tets.shape[0]):
        tet_surface_origin[j] = []

    for i in range(surface_tet_faces.shape[0]):
        face = [surface_tet_faces[i][0], surface_tet_faces[i][1], surface_tet_faces[i][2]]
        face.sort()
        face_str = str(face[0]) + "+" + str(face[1]) + "+" + str(face[2])
        surface_adj_tet[i] = surface_in_tet_map[face_str]
        assert len(surface_adj_tet[i]) > 0
        for tt in surface_in_tet_map[face_str]:
            tet_surface_origin[tt].append(i)

    # for i in range(surface_tet_faces.shape[0]):
    #     face = surface_tet_faces[i]
    #     for j in range(tets.shape[0]):
    #         if all (v in tets[j] for v in face):
    #             surface_adj_tet[i].append(j)
    #             tet_surface_origin[j].append(i)
    #         if len(surface_adj_tet[i]) >= 2:
    #             break
    #     if len(surface_adj_tet[i]) >= 2:
    #         continue  

    print("computed tet->surface and surface->tet mapping.")
    
    # do simplicial embedding
    print("Doing simplicial embedding ...")
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
            v_new = (vertices[tets[i][0]] + vertices[tets[i][1]] + vertices[tets[i][2]] + vertices[tets[i][3]]) / 4.0
            tets_vertices_regular.append(v_new.tolist())

            # create new tets
            new_t_ids = [i, len(tets_regular), len(tets_regular) + 1, len(tets_regular) + 2]
            tets_regular[i] = [new_v_id, tets[i][1], tets[i][2], tets[i][3]]
            tets_regular.append([tets[i][0], new_v_id, tets[i][2], tets[i][3]])
            tets_regular.append([tets[i][0], tets[i][1], new_v_id, tets[i][3]])
            tets_regular.append([tets[i][0], tets[i][1], tets[i][2], new_v_id])

            # reset tet surface mapping
            for tid in new_t_ids:
                tet_surface[tid] = []

            # update map
            for f in fs:
                surface_adj_tet[f].remove(i)
                for tid in new_t_ids:
                    if (face_in_tet(surface_tet_faces[f], tets_regular[tid])):
                        surface_adj_tet[f].append(tid)
                        tet_surface[tid] = [f]
                        break

    tets_regular = np.array(tets_regular)
    tets_vertices_regular = np.array(tets_vertices_regular)

    print("Done simplicial embedding. Splitted {} tets".format(simplicial_embedding_cnt))

    ####################################################
    #             Call Parametrization Code            #
    ####################################################
    print("Calling parametrization code")
    # para_command = path_to_para_exe + " --mesh " + workspace_path + "embedded_surface.obj --fit_field --output " + workspace_path
    para_command = path_to_para_exe + " --mesh " + workspace_path + "embedded_surface.obj --fit_field"
    print(para_command.split())
    subprocess.run(para_command.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("here")


    ####################################################
    #                     Para Split                   #
    ####################################################
    # get para in and out mapping
    print("Doing parametrization input to output mapping ...")
    para_out_file = workspace_path + "parameterized_mesh.obj" # the file name para code should generate
    para_out_v, para_out_tc, _, para_out_f, para_out_ftc, _ = igl.read_obj(para_out_file)

    print("after para #v: {0}, before para #v: {1}".format(para_in_v.shape[0], para_out_v.shape[0]))
    v_thres = para_in_v.shape[0]

    # old face existance
    new_face_ids = []
    for i in range(para_out_f.shape[0]):
        if any(v_out >= v_thres for v_out in para_out_f[i]):
            new_face_ids.append(i)

    deleted_old_fids = []
    used_new_fids = [False for f in para_out_f]

    # face dict existance
    para_in_face_existance = {}
    para_out_face_existance = {}
    # for i in range(para_in_f.shape[0]):
    #     face = [para_in_f[i][0], para_in_f[i][1], para_in_f[i][2]]
    #     face.sort()
    #     face_str = str(face[0]) + "+" + str(face[1]) + "+" + str(face[2])
    #     para_in_face_existance[face_str] = i

    for i in range(para_out_f.shape[0]):
        face = [para_out_f[i][0], para_out_f[i][1], para_out_f[i][2]]
        face.sort()
        face_str = str(face[0]) + "+" + str(face[1]) + "+" + str(face[2])
        para_out_face_existance[face_str] = i

    para_in_to_out_face_mapping = {}

    for i in range(para_in_f.shape[0]):
        face = [para_in_f[i][0], para_in_f[i][1], para_in_f[i][2]]
        face.sort()
        face_str = str(face[0]) + "+" + str(face[1]) + "+" + str(face[2])
        if face_str in para_out_face_existance:
            para_in_to_out_face_mapping[i] = [para_out_face_existance[face_str]]
        else:
            deleted_old_fids.append(i)

    # for i in range(para_in_f.shape[0]):
    #     found = False
    #     for j in range(para_out_f.shape[0]):
    #         if used_new_fids[j]:
    #             continue
    #         if face_equal(para_in_f[i], para_out_f[j]):
    #             found = True
    #             used_new_fids[j] = True
    #             para_in_to_out_face_mapping[i] = [j]
    #     if not found:
    #         deleted_old_fids.append(i)

    print("Done parametrization mapping.")

    # match new faces to old faces
    print("Compute old -> new face containment")
    for f_out in new_face_ids:
        bc = (para_out_v[para_out_f[f_out][0]] + para_out_v[para_out_f[f_out][1]] + para_out_v[para_out_f[f_out][2]]) / 3.0
        found = False
        for f_in in deleted_old_fids:
            if on_tri(para_in_v[para_in_f[f_in][0]], para_in_v[para_in_f[f_in][1]], para_in_v[para_in_f[f_in][2]], bc):
                if f_in in para_in_to_out_face_mapping:
                    para_in_to_out_face_mapping[f_in].append(f_out)
                else:
                    para_in_to_out_face_mapping[f_in] = [f_out]
                found = True
                break
        assert found # must find an f_in that contains f_out

    tet_after_para_vertices = copy.deepcopy(tets_vertices_regular.tolist())
    tet_after_para_tets = copy.deepcopy(tets_regular.tolist())

    # add new vertices to tet mesh
    print("Adding parametrized new vertices to tetmesh ... ")
    para_out_v_to_tet_v_map = copy.deepcopy(para_in_v_to_tet_v_map)
    for i in range(para_in_v.shape[0], para_out_v.shape[0]):
        para_out_v_to_tet_v_map[i] = len(tet_after_para_vertices)
        tet_after_para_vertices.append(para_out_v[i].tolist())

    # para_out to tet_out surface mappings
    surface_adj_tet_para_out = {}
    tet_surface_para_out = {}

    print("Splitting tets according to para output ... ")
    # update unsplitted faces
    for f_in in para_in_to_out_face_mapping:
        if f_in not in deleted_old_fids:
            surface_adj_tet_para_out[para_in_to_out_face_mapping[f_in][0]] = surface_adj_tet[f_in]
            for tid in surface_adj_tet_para_out[para_in_to_out_face_mapping[f_in][0]]:
                tet_surface_para_out[tid] = [para_in_to_out_face_mapping[f_in][0]]

    # split corresponding tets
    for f_in in deleted_old_fids:
        f_vs = surface_tet_faces[f_in] # vid in tet regular index
        adj_tets = surface_adj_tet[f_in]
        for t in adj_tets:
            t_vs = tets_regular[t]
            
            # get local ids for f_vs and other point
            local_ids = [-1,-1,-1,-1]
            for i in range(3):
                for j in range(4):
                    if f_vs[i] == t_vs[j]:
                        local_ids[j] = i
                        break
            assert local_ids.count(-1) == 1

            new_tets = []
            assert len(para_in_to_out_face_mapping[f_in]) > 1
            for f_out in para_in_to_out_face_mapping[f_in]:
                f_out_vs = para_out_f[f_out]
                f_out_vs_tet_base = [para_out_v_to_tet_v_map[vid] for vid in f_out_vs]
                new_tet_vs = copy.deepcopy(t_vs)
                for i in range(4):
                    if local_ids[i] != -1:
                        new_tet_vs[i] = f_out_vs_tet_base[local_ids[i]]
                new_tets.append(new_tet_vs)

            # update tet connectivity
            assert len(new_tets) > 1
            tet_after_para_tets[t] = new_tets[0]
            first_split_face = para_in_to_out_face_mapping[f_in][0]
            if first_split_face not in surface_adj_tet_para_out:
                surface_adj_tet_para_out[first_split_face] = [t]
            else:
                surface_adj_tet_para_out[first_split_face].append(t)
            tet_surface_para_out[t] = [first_split_face]

            for i in range(1, len(new_tets)):
                new_tid = len(tet_after_para_tets)
                # print(new_tid)
                tet_after_para_tets.append(new_tets[i])
                split_face = para_in_to_out_face_mapping[f_in][i]
                if split_face not in surface_adj_tet_para_out:
                    surface_adj_tet_para_out[split_face] = [new_tid]
                else:
                    surface_adj_tet_para_out[split_face].append(new_tid)
                tet_surface_para_out[new_tid] = [split_face]
    print("Done Para Split.")

    ####################################################
    #                     Face Split                   #
    ####################################################
    print("Face splitting ...")
    # face split
    tet_after_face_split_tets = []
    tet_after_face_split_vertices = copy.deepcopy(tet_after_para_vertices)

    # fid contains vid
    face_split_f_to_tet_v_map = {}

    surface_tet_cnt = 0
    visited_list = []

    for tid in range(len(tet_after_para_tets)):
        # non surface case
        if tid not in tet_surface_para_out:
            tet_after_face_split_tets.append(tet_after_para_tets[tid])
            continue

        surface_tet_cnt += 1
        visited_list.append(tid)
        # print(surface_tet_cnt)

        # surface case
        t_sf = tet_surface_para_out[tid][0]
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
        local_ids = [-1,-1,-1,-1]
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
            tet_after_face_split_tets.append(new_t)

    print("Done Face Split.")
    # save tetmesh to msh, use gmsh to create high order nodes
    tet_points_after_face_split = np.array(tet_after_face_split_vertices)
    tet_cells_after_face_split = [('tetra', np.array(tet_after_face_split_tets))]
    tetmesh_after_face_split = mio.Mesh(tet_points_after_face_split, tet_cells_after_face_split)
    tetmesh_after_face_split.write(workspace_path + "tetmesh_after_face_split.msh", file_format='gmsh')

    ####################################################
    #                     Call Gmsh                    #
    ####################################################
    print("Calling Gmsh ... ")
    gmsh.initialize()
    gmsh.open(workspace_path + "tetmesh_after_face_split.msh")
    gmsh.model.mesh.setOrder(3)
    gmsh.write(workspace_path + "tetmesh_after_face_split_high_order_tet.msh")

    ####################################################
    #                   Call CT Code                   #
    ####################################################

    print("Calling Clough Tocher code")
    ct_command = path_to_ct_exe + " --input " + workspace_path + "parameterized_mesh.obj -o CT"
    subprocess.run(ct_command.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    ####################################################
    #          Map tri langrange nodes to tet          #
    ####################################################

    ct_interpolants_file = "CT_from_lagrange_nodes.msh"
    ct_input_v_to_output_v_file = "CT_from_lagrange_nodes_input_v_to_output_v_map.txt"

    print("Doing high order tri to tet mapping ...")
    # assume converted to third order in gmsh
    tetmesh_high_order = mio.read(workspace_path + "tetmesh_after_face_split_high_order_tet.msh")
    # assume ct interpolant
    surface_high_order = mio.read(workspace_path + ct_interpolants_file)

    tetmesh_high_order_conn = tetmesh_high_order.cells[0].data
    surface_high_order_conn = surface_high_order.cells[0].data

    surface_high_order_vertices = surface_high_order.points

    # input surface to output surface v mapping
    # surface_mapping_file = "input_v_to_output_v_map.txt"
    surface_input_to_output_v_map = {}
    surface_output_to_input_v_map = {}
    with open(workspace_path + ct_input_v_to_output_v_file, 'r') as file:
        for line in file:
            values = line.split()
            surface_input_to_output_v_map[int(values[0])] = int(values[1])
            surface_output_to_input_v_map[int(values[1])] = int(values[0])

    surface_high_order_conn_with_input_v_idx = copy.deepcopy(surface_high_order_conn)
    surface_high_order_vertices_with_input_v_idx = copy.deepcopy(surface_high_order_vertices)

    for i in range(surface_high_order_conn_with_input_v_idx.shape[0]):
        for j in range(2):
            # print(surface_high_order_conn_with_input_v_idx[i][j])
            surface_high_order_conn_with_input_v_idx[i][j] = surface_output_to_input_v_map[surface_high_order_conn_with_input_v_idx[i][j]]

    for i in range(surface_high_order_vertices_with_input_v_idx.shape[0]):
        if i in surface_output_to_input_v_map:
            surface_high_order_vertices_with_input_v_idx[surface_output_to_input_v_map[i]] = surface_high_order_vertices[i]

    # map the barycenter vertex of each input tri to tet
    assert surface_high_order_conn_with_input_v_idx.shape[0] % 3 == 0
    bc_surface_to_tet_map = {}
    bc_tet_to_surface_map = {}

    for i in range(surface_high_order_conn_with_input_v_idx.shape[0] // 3):
        tet_split_vid = face_split_f_to_tet_v_map[i]
        bc_surface_to_tet_map[surface_high_order_conn_with_input_v_idx[i*3+0][2]] = tet_split_vid
        bc_tet_to_surface_map[tet_split_vid] = surface_high_order_conn_with_input_v_idx[i*3+0][2]

    print("breaking done high order tet conn into face/edge to vertex mappings")
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
        if str([tet[0], tet[1]]) in tet_edge_to_vertices:
            assert tet_edge_to_vertices[str([tet[0], tet[1]])] == [tet[4], tet[5]]
        else:
            tet_edge_to_vertices[str([tet[0], tet[1]])] = [tet[4], tet[5]]
            tet_edge_to_vertices[str([tet[1], tet[0]])] = [tet[5], tet[4]]

        if str([tet[1], tet[2]]) in tet_edge_to_vertices:
            assert tet_edge_to_vertices[str([tet[1], tet[2]])] == [tet[6], tet[7]]
        else:
            tet_edge_to_vertices[str([tet[1], tet[2]])] = [tet[6], tet[7]]
            tet_edge_to_vertices[str([tet[2], tet[1]])] = [tet[7], tet[6]]
    
        if str([tet[2], tet[0]]) in tet_edge_to_vertices:
            assert tet_edge_to_vertices[str([tet[2], tet[0]])] == [tet[8], tet[9]]
        else:
            tet_edge_to_vertices[str([tet[2], tet[0]])] = [tet[8], tet[9]]
            tet_edge_to_vertices[str([tet[0], tet[2]])] = [tet[9], tet[8]]
        
        if str([tet[3], tet[0]]) in tet_edge_to_vertices:
            assert tet_edge_to_vertices[str([tet[3], tet[0]])] == [tet[10], tet[11]]
        else:
            tet_edge_to_vertices[str([tet[3], tet[0]])] = [tet[10], tet[11]]
            tet_edge_to_vertices[str([tet[0], tet[3]])] = [tet[11], tet[10]]

        if str([tet[3], tet[2]]) in tet_edge_to_vertices:
            assert tet_edge_to_vertices[str([tet[3], tet[2]])] == [tet[12], tet[13]]
        else:
            tet_edge_to_vertices[str([tet[3], tet[2]])] = [tet[12], tet[13]]
            tet_edge_to_vertices[str([tet[2], tet[3]])] = [tet[13], tet[12]]
        
        if str([tet[3], tet[1]]) in tet_edge_to_vertices:
            assert tet_edge_to_vertices[str([tet[3], tet[1]])] == [tet[14], tet[15]]
        else:
            tet_edge_to_vertices[str([tet[3], tet[1]])] = [tet[14], tet[15]]
            tet_edge_to_vertices[str([tet[1], tet[3]])] = [tet[15], tet[14]]

        f012 = [tet[0], tet[1], tet[2]]
        f013 = [tet[0], tet[1], tet[3]]
        f023 = [tet[0], tet[2], tet[3]]
        f123 = [tet[1], tet[2], tet[3]]
        f012.sort()
        f013.sort()
        f023.sort()
        f123.sort()

        tet_face_to_vertices[str(f012)] = tet[16]
        tet_face_to_vertices[str(f013)] = tet[17]
        tet_face_to_vertices[str(f023)] = tet[18]
        tet_face_to_vertices[str(f123)] = tet[19]
    
    print("constructing tri <-> tet v mappings")
    # map high order tri vertices to tet vertices
    tri_to_tet_high_order_v_map = {}
    tet_to_tri_high_order_v_map = {}

    for tri in surface_high_order_conn_with_input_v_idx:
        # print(tri)
        vs = [para_out_v_to_tet_v_map[tri[0]], para_out_v_to_tet_v_map[tri[1]], bc_surface_to_tet_map[tri[2]]] # vertices in tet idx
        face = [vs[0], vs[1], vs[2]] 
        face.sort()
        face = str(face)

        e01 = str([vs[0], vs[1]])
        e12 = str([vs[1], vs[2]])
        e20 = str([vs[2], vs[0]])

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
        curved_tet_vertices[key] = surface_high_order_vertices_with_input_v_idx[tet_to_tri_high_order_v_map[key]]

    # write files
    print("writing outputs ...")
    # TODO: file names
    curved_tet_file_name = output_name + "_curved_tetmesh.msh"  
    linear_tet_file_name = output_name + "_initial_tetmesh.msh"  
    displacement_file_name = output_name + "_displacements.txt"
    tri_to_tet_index_mapping_file_name = output_name + "_tri_to_tet_v_map.txt"

    curved_points = curved_tet_vertices
    curved_cells = [('tetra20', curved_tet_conn)]

    curved_tetmesh = mio.Mesh(curved_points, curved_cells)
    curved_tetmesh.write(workspace_path + curved_tet_file_name, file_format='gmsh')

    linear_tetmesh = mio.Mesh(tetmesh_high_order.points, tetmesh_high_order.cells)
    linear_tetmesh.write(workspace_path + linear_tet_file_name, file_format='gmsh')

    delta_mesh_points = curved_tetmesh.points - linear_tetmesh.points
    delta_mesh_cells = []
    delta_mesh = mio.Mesh(delta_mesh_points, curved_cells)

    with open(workspace_path + displacement_file_name, 'w') as file:
        for key in tet_to_tri_high_order_v_map:
            file.write(f'{key} {delta_mesh_points[key][0]} {delta_mesh_points[key][1]} {delta_mesh_points[key][2]}\n')

    # write the tri to tet vertex map
    tri_to_tet_idx_file_correct = open(workspace_path + tri_to_tet_index_mapping_file_name, "w")

    tri_to_tet_constraint_nodes_v_map = {}

    for i in range(len(tri_to_tet_high_order_v_map.keys())):
        if i in surface_input_to_output_v_map.keys():
            tri_to_tet_constraint_nodes_v_map[surface_input_to_output_v_map[i]] = tri_to_tet_high_order_v_map[i]
        else:
            tri_to_tet_constraint_nodes_v_map[i] = tri_to_tet_high_order_v_map[i]

    for i in range(len(tri_to_tet_high_order_v_map.keys())):
        tri_to_tet_idx_file_correct.write(str(tri_to_tet_constraint_nodes_v_map[i])+"\n")

    tri_to_tet_idx_file_correct.close()


    ####################################################
    #           Constuct Constraint Matrix             #
    ####################################################
    print("constructing full hard constraint matrix ...")

    interior_matix = np.loadtxt('CT_interior_constraint_matrix.txt') 
    edge_end_point_matrix = np.loadtxt('CT_edge_endpoint_constraint_matrix_eliminated.txt') 
    edge_mid_point_matrix = np.loadtxt('CT_edge_midpoint_constraint_matrix.txt')

    full_matrix = np.concatenate((interior_matix, edge_end_point_matrix, edge_mid_point_matrix))

    local2global = np.loadtxt(workspace_path + tri_to_tet_index_mapping_file_name).astype(np.int32)
    A = full_matrix
    m = mio.read(workspace_path + linear_tet_file_name)
    v = m.points
    b = -(A @ v[local2global, :])

    with h5py.File(workspace_path  + "CT_full_constraint_matrix.hdf5", 'w') as f:
        f.create_dataset("local2global", data=local2global.astype(np.int32))
        f.create_dataset("A", data=A)
        f.create_dataset("b", data=b)

    print("constructing soft constraint matrix ...")

    # A_1 = L u  b_1 = -L x_0
    # A_2 = I    b_2 = x0 - xtrg (for now xtrg can be zero to run some experiments)

    lap_conn_file = "CT_bilaplacian_nodes.obj"
    v_lap, _, _, f_lap, _, _ = igl.read_obj(lap_conn_file)

    L_w = igl.cotmatrix(v_lap, f_lap)
    M = igl.massmatrix(v_lap, f_lap)

    # assemble M_inv
    M_inv_rows = np.array([i for i in range(M.shape[0])])
    M_inv_cols = np.array([i for i in range(M.shape[1])])
    M_inv_data = np.array([1.0/M[i,i] for i in M_inv_rows])
    M_size = len(M_inv_cols)
    M_inv = sparse.csc_matrix((M_inv_data, (M_inv_rows, M_inv_cols)), shape=(M_size, M_size))

    L = M_inv @ L_w

    A_1 = L
    b_1 = -L @ v[local2global, :]
    A_2 = sparse.identity(len(local2global))
    b_2 = v[local2global, :] # TODO: add xtrg

    A_1_p = A_1.tocoo(True)
    A_2_p = A_2.tocoo(True)

    with h5py.File("soft_1.hdf5", 'w') as file:
        file.create_dataset("b", data=b_1)
        file.create_dataset("A_triplets/values", data=A_1_p.data)
        file.create_dataset("A_triplets/cols", data=A_1_p.col)
        file.create_dataset("A_triplets/rows", data=A_1_p.row)
        file.create_dataset("local2global", data=local2global.astype(np.int32))

    with h5py.File("soft_2.hdf5", 'w') as file:
        file.create_dataset("b", data=b_2)
        file.create_dataset("A_triplets/values", data=A_2_p.data)
        file.create_dataset("A_triplets/cols", data=A_2_p.col)
        file.create_dataset("A_triplets/rows", data=A_2_p.row)
        file.create_dataset("local2global", data=local2global.astype(np.int32))

    # ####################################################
    # #            create json  for Polyfem              #
    # ####################################################

    c_json = {'space': {'discr_order': 3}, 'geometry': [{'mesh': output_name + '_initial_tetmesh.msh', 'volume_selection': 1, 'surface_selection': 1}], 'constraints': {'hard': ['CT_full_constraint_matrix.hdf5'], 'soft': [{'weight': 10000.0, 'data': 'soft_1.hdf5'}, {'weight': 10000.0, 'data': 'soft_2.hdf5'}]}, 'materials': [{'id': 1, 'type': 'NeoHookean', 'E': 20000000.0, 'nu': 0.3}], 'solver': {'nonlinear': {'x_delta': 1e-10, 'solver': 'Newton', 'grad_norm': 1e-08, 'advanced': {'f_delta': 1e-10}}, 'augmented_lagrangian': {'initial_weight': 100000000.0}}, 'boundary_conditions': {'dirichlet_boundary': {'id': 1, 'value': [0, 0, 0]}}, 'output': {'paraview': {'file_name': 'sim3d.vtu', 'surface': True, 'wireframe': True, 'points': True, 'options': {'material': True, 'force_high_order': True}, 'vismesh_rel_area': 1e-05}}}
    with open('constraints.json', 'w') as f:
        json.dump(c_json, f)

    # ####################################################
    # #                    Call Polyfem                  #
    # ####################################################
    
    # print("Calling Polyfem")
    polyfem_command = path_to_polyfem_exe + " -j " + workspace_path + "constraints.json"
    subprocess.run(polyfem_command.split(' '), stdout=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)








    





                    
                    

                    



                                