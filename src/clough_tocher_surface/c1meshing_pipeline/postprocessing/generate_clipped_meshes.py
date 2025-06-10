import igl
import meshio as mio
import numpy as np
import scipy
import math
import copy
import sys


def sample1(n):
    v = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        #
        [1/3, 0],
        [2/3, 0],

        [2/3, 1/3],
        [1/3, 2/3],

        [0, 2/3],
        [0, 1/3],

        [1/3, 1/3]
    ])

    f = np.array([
        [0, 3, 8],
        [3, 9, 8],
        [3, 4, 9],
        [4, 5, 9],
        [4, 1, 5],
        [8, 9, 7],
        [9, 6, 7],
        [9, 5, 6],
        [7, 6, 2]
    ])

    v, f = igl.upsample(v, f, n)

    return v, f


def sample(n):
    V = np.zeros((n*n, 2))
    F = np.zeros((2*(n-1)*(n-1), 3), dtype=int)
    delta = 1. / (n - 1)
    map = np.full((n, n), -1, dtype=int)
    index = 0
    for i in range(n):
        for j in range(n):
            if i + j >= n:
                continue
            map[i, j] = index
            V[index] = [i * delta, j * delta]
            index += 1
    V = V[:index]
    index = 0
    for i in range(n - 1):
        for j in range(n - 1):
            if map[i, j] >= 0 and map[i+1, j] >= 0 and map[i, j+1] >= 0:
                F[index] = [map[i, j], map[i+1, j], map[i, j+1]]
                index += 1
            if map[i+1, j] >= 0 and map[i+1, j+1] >= 0 and map[i, j+1] >= 0:
                F[index] = [map[i+1, j], map[i+1, j+1], map[i, j+1]]
                index += 1
    F = F[:index]
    return V, F


def lagr0(x, y):
    helper_0 = pow(x, 2)
    helper_1 = pow(y, 2)
    result_0 = -27.0 / 2.0 * helper_0 * y + 9 * helper_0 - 27.0 / 2.0 * helper_1 * x + 9 * helper_1 - \
        9.0 / 2.0 * pow(x, 3) + 18 * x * y - 11.0 / 2.0 * x - \
        9.0 / 2.0 * pow(y, 3) - 11.0 / 2.0 * y + 1

    return result_0


def lagr1(x, y):
    result_0 = (1.0 / 2.0) * x * (9 * pow(x, 2) - 9 * x + 2)
    return result_0


def lagr2(x, y):
    result_0 = (1.0 / 2.0) * y * (9 * pow(y, 2) - 9 * y + 2)
    return result_0


def lagr3(x, y):
    result_0 = (9.0 / 2.0) * x * (x + y - 1) * (3 * x + 3 * y - 2)
    return result_0


def lagr4(x, y):
    result_0 = -9.0 / 2.0 * x * (3 * pow(x, 2) + 3 * x * y - 4 * x - y + 1)
    return result_0


def lagr5(x, y):
    result_0 = (9.0 / 2.0) * x * y * (3 * x - 1)
    return result_0


def lagr6(x, y):
    result_0 = (9.0 / 2.0) * x * y * (3 * y - 1)
    return result_0


def lagr7(x, y):
    result_0 = -9.0 / 2.0 * y * (3 * x * y - x + 3 * pow(y, 2) - 4 * y + 1)
    return result_0


def lagr8(x, y):
    result_0 = (9.0 / 2.0) * y * (x + y - 1) * (3 * x + 3 * y - 2)
    return result_0


def lagr9(x, y):
    result_0 = -27 * x * y * (x + y - 1)
    return result_0


def eval_lagr(p, nodes):
    lagrs = [lagr0, lagr1, lagr2, lagr3, lagr4,
             lagr5, lagr6, lagr7, lagr8, lagr9]

    x = p[:, 0]
    y = p[:, 1]

    res = np.zeros((p.shape[0], nodes.shape[1]))

    for i, n in enumerate(nodes):
        res += lagrs[i](x, y)[:, None]*n

    return res

# upsample


def upsample_mesh(upsample_factor, ho_nodes, triangle10):
    v_upsample_local, f_upsample_local = sample1(upsample_factor)

    v_upsample = []
    f_upsample = []
    offset = 0

    tris = triangle10
    for i, tt in enumerate(tris):
        nodes = ho_nodes[tt]
        newv = eval_lagr(v_upsample_local, nodes)
        v_upsample.append(newv)
        f_upsample.append(f_upsample_local+offset)
        offset += newv.shape[0]
        # if i > 1:
        #     break

    v_upsample = np.row_stack(v_upsample)
    f_upsample = np.row_stack(f_upsample)

    return v_upsample, f_upsample


def berstein_coeff_cubic(i, j, k):
    return math.factorial(3)/(math.factorial(i) * math.factorial(j) * math.factorial(k))


def berstein_monomial(ijk, uv):
    i, j, k = ijk
    u, v = uv
    w = 1.0 - u - v
    return pow(u, i) * pow(v, j) * pow(w, k)


def get_b2l_mat():
    barycoords = np.array([
        [0, 0],
        [1, 0],
        [0, 1],

        [1/3, 0],
        [2/3, 0],

        [2/3, 1/3],
        [1/3, 2/3],

        [0, 2/3],
        [0, 1/3],

        [1/3, 1/3]
    ])

    ijk = np.array([
        [0, 0, 3],
        [3, 0, 0],
        [0, 3, 0],
        [1, 0, 2],
        [2, 0, 1],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [0, 1, 2],
        [1, 1, 1]
    ])

    b2l_mat = np.zeros((10, 10))

    for row in range(10):
        for col in range(10):
            b2l_mat[row][col] = berstein_coeff_cubic(
                ijk[col][0], ijk[col][1], ijk[col][2]) * berstein_monomial(ijk[col], barycoords[row])

    return b2l_mat

# construct all tet cubic faces


def process_clipped_tetmesh(tv, t_clip, upsample_factor, eps, output_name):
    face_map_clip = {}
    for tet20 in t_clip:
        face_map_clip[str(tet20[0]) + "+"+str(tet20[1])+"+"+str(tet20[2])
                      ] = tet20[[0, 1, 2, 4, 5, 6, 7, 8, 9, 16]]  # 012
        face_map_clip[str(tet20[1]) + "+"+str(tet20[0])+"+"+str(tet20[2])
                      ] = tet20[[1, 0, 2, 5, 4, 9, 8, 7, 6, 16]]  # 102
        face_map_clip[str(tet20[0]) + "+"+str(tet20[1])+"+"+str(tet20[3])
                      ] = tet20[[0, 1, 3, 4, 5, 15, 14, 10, 11, 17]]  # 013
        face_map_clip[str(tet20[1]) + "+"+str(tet20[0])+"+"+str(tet20[3])
                      ] = tet20[[1, 0, 3, 5, 4, 11, 10, 14, 15, 17]]  # 103
        face_map_clip[str(tet20[0]) + "+"+str(tet20[2])+"+"+str(tet20[3])
                      ] = tet20[[0, 2, 3, 9, 8, 13, 12, 10, 11, 18]]  # 023
        face_map_clip[str(tet20[2]) + "+"+str(tet20[0])+"+"+str(tet20[3])
                      ] = tet20[[2, 0, 3, 8, 9, 11, 10, 12, 13, 18]]  # 203
        face_map_clip[str(tet20[1]) + "+"+str(tet20[2])+"+"+str(tet20[3])
                      ] = tet20[[1, 2, 3, 6, 7, 13, 12, 14, 15, 19]]  # 123
        face_map_clip[str(tet20[2]) + "+"+str(tet20[1])+"+"+str(tet20[3])
                      ] = tet20[[2, 1, 3, 7, 6, 15, 14, 12, 13, 19]]  # 123

    boundary_faces_clip = igl.boundary_facets(t_clip[:, :4])

    cubic_faces_clip = []
    for f in boundary_faces_clip:
        f012 = str(f[0]) + "+" + str(f[1]) + "+" + str(f[2])
        f120 = str(f[1]) + "+" + str(f[2]) + "+" + str(f[0])
        f201 = str(f[2]) + "+" + str(f[0]) + "+" + str(f[1])
        if f012 in face_map_clip:
            cubic_faces_clip.append(face_map_clip[f012])
        elif f120 in face_map_clip:
            cubic_faces_clip.append(face_map_clip[f120])
        elif f201 in face_map_clip:
            cubic_faces_clip.append(face_map_clip[f201])
    cubic_faces_clip = np.array(cubic_faces_clip)

    cubic_clipped_sf_mesh = mio.Mesh(tv, [('triangle10', cubic_faces_clip)])
    cubic_clipped_sf_mesh.write(
        output_name + "_cubic_surface_clipped.msh", file_format="gmsh")

    # upsample clipped
    v_up_clipped, f_up_clipped = upsample_mesh(
        upsample_factor, cubic_clipped_sf_mesh.points, cubic_clipped_sf_mesh.cells_dict['triangle10'])

    v_up_clip_clean, _, _, f_up_clip_clean = igl.remove_duplicate_vertices(
        v_up_clipped, f_up_clipped, eps)
    [v_up_clip_clean, f_up_clip_clean, _, _] = igl.remove_unreferenced(
        v_up_clip_clean, f_up_clip_clean)

    upsampled_mesh_clipped = mio.Mesh(
        v_up_clip_clean, [('triangle', f_up_clip_clean)])
    upsampled_mesh_clipped.write(output_name + ".obj")

    # extract edges
    upsampled_edges_clipped = igl.boundary_facets(f_up_clipped)
    degnerate_tris_clip = np.array(
        [[e[0], e[1], e[0]] for e in upsampled_edges_clipped])
    v_edges, _, _, f_edges = igl.remove_duplicate_vertices(
        v_up_clipped, degnerate_tris_clip, eps)
    [v_edges, f_edges, _, _] = igl.remove_unreferenced(v_edges, f_edges)

    processed_edge = {}
    with open(output_name + "_edges.obj", "w") as edge_file:
        for vv in v_edges:
            edge_file.write("v {} {} {}\n".format(vv[0], vv[1], vv[2]))
        for edge in f_edges[:, :2]:
            if str(edge[1]) + "+" + str(edge[0]) not in processed_edge:
                edge_file.write("l {} {}\n".format(edge[0]+1, edge[1]+1))
                processed_edge[str(edge[0]) + "+" + str(edge[1])] = True

    return cubic_faces_clip


def get_upsampled_obj_mesh(cubic_sf_mesh, suffix):
    v_up_inter, f_up_inter = upsample_mesh(
        upsample_factor, cubic_sf_mesh.points, cubic_sf_mesh.cells_dict['triangle10'])

    v_up_inter_clean, _, _, f_up_inter_clean = igl.remove_duplicate_vertices(
        v_up_inter, f_up_inter, eps)
    [v_up_inter_clean, f_up_inter_clean, _, _] = igl.remove_unreferenced(
        v_up_inter_clean, f_up_inter_clean)

    upsampled_mesh_clipped = mio.Mesh(
        v_up_inter_clean, [('triangle', f_up_inter_clean)])
    upsampled_mesh_clipped.write(output_name + suffix + ".obj")

    # extract edges
    upsampled_edges_inter = igl.boundary_facets(f_up_inter)
    degnerate_tris_inter = np.array(
        [[e[0], e[1], e[0]] for e in upsampled_edges_inter])
    v_edges, _, _, f_edges = igl.remove_duplicate_vertices(
        v_up_inter, degnerate_tris_inter, eps)
    [v_edges, f_edges, _, _] = igl.remove_unreferenced(v_edges, f_edges)

    processed_edge = {}
    with open(output_name + suffix + "_edges.obj", "w") as edge_file:
        for vv in v_edges:
            edge_file.write("v {} {} {}\n".format(vv[0], vv[1], vv[2]))
        for edge in f_edges[:, :2]:
            if str(edge[1]) + "+" + str(edge[0]) not in processed_edge:
                edge_file.write("l {} {}\n".format(edge[0]+1, edge[1]+1))
                processed_edge[str(edge[0]) + "+" + str(edge[1])] = True


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 12:
        print("args unmatch!")
        print("needed: <init_file.msh> <solution.txt> <inside.msh> <sf_mesh_file.msh> <local2global_file.txt> <winding_number_file.txt> <output_name> <upsample_factor> <clip_axis> <clip_ratio> <eps>")

    init_file = args[1]
    solutino_file = args[2]
    inside_file = args[3]
    sf_mesh_file = args[4]
    winding_file = args[5]
    local2global_file = args[6]
    output_name = args[7]
    upsample_factor = int(args[8])  # better <= 2, otherwise super dense
    clip_axis = int(args[9])  # 0 for x,  1 for y,  2 for z
    clip_ratio = float(args[10])  # [min, min + ratio * (max - min)]
    # eps: please use 1e-14 for default, for remove duplicates
    eps = float(args[11])

    # read initial mesh and solution
    init_mesh = mio.read(init_file)
    v_init = init_mesh.points
    t_init = init_mesh.cells_dict['tetra20']
    solution = np.loadtxt(solutino_file)

    # get result beizer control points
    v_res = v_init + solution

    # convert to lagrange
    v_lagrange = np.zeros(v_res.shape)

    b2l_mat = get_b2l_mat()

    for tet in t_init:
        fs = [np.array(v_res[tet[[0, 1, 2, 4, 5, 6, 7, 8, 9, 16]]]),
              np.array(v_res[tet[[0, 1, 3, 4, 5, 15, 14, 10, 11, 17]]]),
              np.array(v_res[tet[[0, 2, 3, 9, 8, 13, 12, 10, 11, 18]]]),
              np.array(v_res[tet[[1, 2, 3, 6, 7, 13, 12, 14, 15, 19]]])]

        fs_ids = [np.array(tet[[0, 1, 2, 4, 5, 6, 7, 8, 9, 16]]),
                  np.array(tet[[0, 1, 3, 4, 5, 15, 14, 10, 11, 17]]),
                  np.array(tet[[0, 2, 3, 9, 8, 13, 12, 10, 11, 18]]),
                  np.array(tet[[1, 2, 3, 6, 7, 13, 12, 14, 15, 19]])]

        fs_lag = [b2l_mat @ f for f in fs]

        for i in range(4):
            for j in range(10):
                v_lagrange[fs_ids[i][j]] = fs_lag[i][j]

    # read winding and filter outside
    winding_numbers = np.loadtxt(winding_file)

    t_kept = []
    for i in range(t_init.shape[0]):
        if abs(winding_numbers[i][1]) >= 0.5:
            t_kept.append(t_init[i])

    t_kept = np.array(t_kept)

    # remove unreference
    v_kept_map = {}
    v_kept = []
    for tet in t_kept:
        for vid in tet:
            if vid not in v_kept_map:
                v_kept_map[vid] = len(v_kept)
                v_kept.append(v_lagrange[vid])
            else:
                # do nothing
                continue

    v_kept = np.array(v_kept)
    t_kept_clean = [[v_kept_map[tet[i]] for i in range(20)] for tet in t_kept]

    # write cubic tetmesh
    res_mesh = mio.Mesh(v_kept, [("tetra20", t_kept_clean)])
    res_mesh.write(output_name+".msh", file_format="gmsh")

    # get result surface mesh
    local2global = np.loadtxt(
        local2global_file).astype(np.int32)
    v_sf_lag = v_lagrange[local2global]
    sf_mesh = mio.read(sf_mesh_file)
    res_sf_mesh = mio.Mesh(
        v_sf_lag, [("triangle10", sf_mesh.cells_dict["triangle10"])])
    res_sf_mesh.write(output_name + "_sf.msh", file_format="gmsh")

    # upsample surface mesh
    v_upsample, f_upsample = upsample_mesh(
        upsample_factor, res_sf_mesh.points, res_sf_mesh.cells_dict['triangle10'])

    v_upsample_stitched, _, _, f_upsample_stitched = igl.remove_duplicate_vertices(
        v_upsample, f_upsample, eps)

    upsampled_mesh = mio.Mesh(v_upsample_stitched, [
                              ('triangle', f_upsample_stitched)])
    upsampled_mesh.write(output_name + "_upsampled_mesh.obj")

    # extract edges
    upsampled_edges = igl.boundary_facets(f_upsample)
    degnerate_tris = np.array([[e[0], e[1], e[0]] for e in upsampled_edges])
    v_edges, _, _, f_edges = igl.remove_duplicate_vertices(
        v_upsample, degnerate_tris, eps)
    [v_edges, f_edges, _, _] = igl.remove_unreferenced(v_edges, f_edges)

    processed_edge = {}
    with open(output_name + "_upsampled_edge.obj", "w") as edge_file:
        for vv in v_edges:
            edge_file.write("v {} {} {}\n".format(vv[0], vv[1], vv[2]))
        for edge in f_edges[:, :2]:
            if str(edge[1]) + "+" + str(edge[0]) not in processed_edge:
                edge_file.write("l {} {}\n".format(edge[0]+1, edge[1]+1))
                processed_edge[str(edge[0]) + "+" + str(edge[1])] = True

    #######################################################################
    ############################# clip mesh ###############################
    #######################################################################

    # get clip plane
    clip_max = np.max(res_mesh.points[:, clip_axis])
    clip_min = np.min(res_mesh.points[:, clip_axis])

    clip_plane = clip_min + clip_ratio * (clip_max - clip_min)

    # get clip/drop tets
    tv = res_mesh.points
    tv_init = res_mesh.points
    t_clip = []
    t_drop = []
    for tet in res_mesh.cells_dict['tetra20']:
        # use init coord to compute centroid, otherwise may not match with internal
        centroid = (tv_init[tet[0]] + tv_init[tet[1]] +
                    tv_init[tet[2]] + tv_init[tet[3]])/4.
        if centroid[clip_axis] > clip_plane:
            t_clip.append(tet)
        else:
            t_drop.append(tet)
    t_clip = np.array(t_clip)
    t_drop = np.array(t_drop)

    cubic_faces_clip = process_clipped_tetmesh(
        tv, t_clip, upsample_factor, eps, output_name + "_clipped")
    cubic_faces_drop = process_clipped_tetmesh(
        tv, t_drop, upsample_factor, eps, output_name + "_dropped")

    clip_face_map = {}

    for f in cubic_faces_clip:
        f_perm = [str(f[0]) + "+" + str(f[1]) + "+" + str(f[2]),
                  str(f[0]) + "+" + str(f[2]) + "+" + str(f[1]),
                  str(f[1]) + "+" + str(f[2]) + "+" + str(f[0]),
                  str(f[1]) + "+" + str(f[0]) + "+" + str(f[2]),
                  str(f[2]) + "+" + str(f[0]) + "+" + str(f[1]),
                  str(f[2]) + "+" + str(f[1]) + "+" + str(f[0])]

        for f_p in f_perm:
            clip_face_map[f_p] = f

    drop_face_map = {}

    for f in cubic_faces_drop:
        f_perm = [str(f[0]) + "+" + str(f[1]) + "+" + str(f[2]),
                  str(f[0]) + "+" + str(f[2]) + "+" + str(f[1]),
                  str(f[1]) + "+" + str(f[2]) + "+" + str(f[0]),
                  str(f[1]) + "+" + str(f[0]) + "+" + str(f[2]),
                  str(f[2]) + "+" + str(f[0]) + "+" + str(f[1]),
                  str(f[2]) + "+" + str(f[1]) + "+" + str(f[0])]

        for f_p in f_perm:
            drop_face_map[f_p] = f

    # get intersection part
    f_intersect = []
    for f in cubic_faces_drop:
        f_p = str(f[0]) + "+" + str(f[1]) + "+" + str(f[2])
        if f_p in clip_face_map:
            f_intersect.append(clip_face_map[f_p])
    f_intersect = np.array(f_intersect)

    cubic_intersect_sf_mesh = mio.Mesh(tv, [('triangle10', f_intersect)])
    cubic_intersect_sf_mesh.write(
        output_name + "_cubic_surface_intersect.msh", file_format="gmsh")

    # get clip/drop sf mesh without intersection part
    f_clip_wo_intersect = []
    for f in cubic_faces_clip:
        f_p = str(f[0]) + "+" + str(f[1]) + "+" + str(f[2])
        if f_p not in drop_face_map:
            f_clip_wo_intersect.append(f)
    f_clip_wo_intersect = np.array(f_clip_wo_intersect)

    clip_wo_intersect_sf_mesh = mio.Mesh(
        tv, [('triangle10', f_clip_wo_intersect)])
    clip_wo_intersect_sf_mesh.write(
        output_name + "_clip_wo_intersect.msh", file_format="gmsh")

    f_drop_wo_intersect = []
    for f in cubic_faces_drop:
        f_p = str(f[0]) + "+" + str(f[1]) + "+" + str(f[2])
        if f_p not in clip_face_map:
            f_drop_wo_intersect.append(f)
    f_drop_wo_intersect = np.array(f_drop_wo_intersect)

    drop_wo_intersect_sf_mesh = mio.Mesh(
        tv, [('triangle10', f_drop_wo_intersect)])
    drop_wo_intersect_sf_mesh.write(
        output_name + "_drop_wo_intersect.msh", file_format="gmsh")

    get_upsampled_obj_mesh(cubic_intersect_sf_mesh, "_intersect")
    get_upsampled_obj_mesh(clip_wo_intersect_sf_mesh, "_clip_wo_intersect")
    get_upsampled_obj_mesh(drop_wo_intersect_sf_mesh, "_drop_wo_intersect")

    #######################################################################
    ############################ inside mesh ##############################
    #######################################################################

    # inside clipped
    try:
        inside_mesh = mio.read(inside_file)
    except:
        print("cannot read init file, may not exist")
        exit()
    tv_in = inside_mesh.points
    tet_kept_in = []
    tet_drop_in = []
    for tet in inside_mesh.cells_dict['tetra']:
        centroid = (tv_in[tet[0]] + tv_in[tet[1]] +
                    tv_in[tet[2]] + tv_in[tet[3]])/4.
        # if centroid[1] > y_mid:
        if centroid[clip_axis] > clip_plane:
            tet_kept_in.append(tet)
        else:
            tet_drop_in.append(tet)
    tet_kept_in = np.array(tet_kept_in)
    tet_drop_in = np.array(tet_drop_in)

    boundary_faces_in = igl.boundary_facets(tet_kept_in)
    [v_in_k, f_in_k, _, _] = igl.remove_unreferenced(tv_in, boundary_faces_in)

    inside_sf_mesh = mio.Mesh(v_in_k, [('triangle', f_in_k)])
    inside_sf_mesh.write(output_name + "_inside_mesh_clip.obj")

    # extract edges
    processed_edge = {}
    with open(output_name + "_inside_edge_clip.obj", "w") as edge_file:
        for vv in inside_sf_mesh.points:
            edge_file.write("v {} {} {}\n".format(vv[0], vv[1], vv[2]))
        for ff in inside_sf_mesh.cells_dict['triangle']:
            if str(ff[1]) + "+" + str(ff[0]) not in processed_edge:
                edge_file.write("l {} {}\n".format(ff[0]+1, ff[1]+1))
                processed_edge[str(ff[0]) + "+" + str(ff[1])] = True
            if str(ff[2]) + "+" + str(ff[1]) not in processed_edge:
                edge_file.write("l {} {}\n".format(ff[1]+1, ff[2]+1))
                processed_edge[str(ff[1]) + "+" + str(ff[2])] = True
            if str(ff[0]) + "+" + str(ff[2]) not in processed_edge:
                edge_file.write("l {} {}\n".format(ff[2]+1, ff[0]+1))
                processed_edge[str(ff[2]) + "+" + str(ff[0])] = True

    boundary_faces_in_drop = igl.boundary_facets(tet_drop_in)
    [v_in_d, f_in_d, _, _] = igl.remove_unreferenced(
        tv_in, boundary_faces_in_drop)

    inside_sf_mesh_d = mio.Mesh(v_in_d, [('triangle', f_in_d)])
    inside_sf_mesh_d.write(output_name + "_inside_mesh_drop.obj")

    # extract edges
    processed_edge = {}
    with open(output_name + "_inside_edge_drop.obj", "w") as edge_file:
        for vv in inside_sf_mesh_d.points:
            edge_file.write("v {} {} {}\n".format(vv[0], vv[1], vv[2]))
        for ff in inside_sf_mesh_d.cells_dict['triangle']:
            if str(ff[1]) + "+" + str(ff[0]) not in processed_edge:
                edge_file.write("l {} {}\n".format(ff[0]+1, ff[1]+1))
                processed_edge[str(ff[0]) + "+" + str(ff[1])] = True
            if str(ff[2]) + "+" + str(ff[1]) not in processed_edge:
                edge_file.write("l {} {}\n".format(ff[1]+1, ff[2]+1))
                processed_edge[str(ff[1]) + "+" + str(ff[2])] = True
            if str(ff[0]) + "+" + str(ff[2]) not in processed_edge:
                edge_file.write("l {} {}\n".format(ff[2]+1, ff[0]+1))
                processed_edge[str(ff[2]) + "+" + str(ff[0])] = True

    # deprecated!!!
    # notice that intersection of inside and intersection of result cannot make up the full clip face!!! i.e. There are face shared by the inside and result, which are not intersections of either part, but need to be visualized
    # use intersection of result + clipped inside instead to get different color!!!

    # # get inside intersect faces
    # inside_clip_face_map = {}

    # for f in boundary_faces_in:
    #     f_perm = [str(f[0]) + "+" + str(f[1]) + "+" + str(f[2]),
    #                 str(f[0]) + "+" + str(f[2]) + "+" + str(f[1]),
    #                 str(f[1]) + "+" + str(f[2]) + "+" + str(f[0]),
    #                 str(f[1]) + "+" + str(f[0]) + "+" + str(f[2]),
    #                 str(f[2]) + "+" + str(f[0]) + "+" + str(f[1]),
    #                 str(f[2]) + "+" + str(f[1]) + "+" + str(f[0])]

    #     for f_p in f_perm:
    #         inside_clip_face_map[f_p] = f

    # inside_f_intersect = []
    # for f in boundary_faces_in_drop:
    #     f_p = str(f[0]) + "+" + str(f[1]) + "+" + str(f[2])
    #     if f_p in inside_clip_face_map:
    #         inside_f_intersect.append(inside_clip_face_map[f_p])
    # inside_f_intersect = np.array(inside_f_intersect)

    # [v_inside_inter, f_inside_inter, _, _] = igl.remove_unreferenced(tv_in, inside_f_intersect)
    # inside_intersect_sf_mesh = mio.Mesh(v_inside_inter, [('triangle', f_inside_inter)])
    # inside_intersect_sf_mesh.write(
    #             output_name + "_inside_intersect.obj")

    # # get inside intersect edges
    # processed_edge = {}
    # with open(output_name + "_inside_intersect_edges.obj", "w") as edge_file:
    #     for vv in inside_intersect_sf_mesh.points:
    #         edge_file.write("v {} {} {}\n".format(vv[0], vv[1], vv[2]))
    #     for ff in inside_intersect_sf_mesh.cells_dict['triangle']:
    #         if str(ff[1]) + "+" + str(ff[0]) not in processed_edge:
    #             edge_file.write("l {} {}\n".format(ff[0]+1, ff[1]+1))
    #             processed_edge[str(ff[0]) + "+" + str(ff[1])] = True
    #         if str(ff[2]) + "+" + str(ff[1]) not in processed_edge:
    #             edge_file.write("l {} {}\n".format(ff[1]+1, ff[2]+1))
    #             processed_edge[str(ff[1]) + "+" + str(ff[2])] = True
    #         if str(ff[0]) + "+" + str(ff[2]) not in processed_edge:
    #             edge_file.write("l {} {}\n".format(ff[2]+1, ff[0]+1))
    #             processed_edge[str(ff[2]) + "+" + str(ff[0])] = True
