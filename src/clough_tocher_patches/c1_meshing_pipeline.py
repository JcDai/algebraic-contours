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
from argparse import ArgumentParser


# check orient3d > 0
def orient3d(aa, bb, cc, dd):
    a = np.array(aa)
    b = np.array(bb)
    c = np.array(cc)
    d = np.array(dd)
    mat = np.zeros([3, 3])
    mat[0, :] = b - a
    mat[1, :] = c - a
    mat[2, :] = d - a

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
    AB_n = AB / np.linalg.norm(AB)
    AC_n = AC / np.linalg.norm(AC)
    AD_n = AD / np.linalg.norm(AD)

    r = AD_n @ np.cross(AB_n, AC_n)
    if abs(r) > eps:
        return False

    # check in shape
    c1 = np.cross(B - A, D - A)
    c2 = np.cross(C - B, D - B)
    c3 = np.cross(A - C, D - C)

    if c1 @ c2 > 0 and c1 @ c3 > 0:
        return True

    return False

def sample1(n):
    v = np.array([
        [0,0],
        [1,0],
        [0,1],
        #
        [1/3,0],
        [2/3,0],

        [2/3,1/3],
        [1/3,2/3],
    
        [0,2/3],
        [0,1/3],

        [1/3,1/3]
    ])

    f = np.array([
        [0,3,8],
        [3,9,8],
        [3,4,9],
        [4,5,9],
        [4,1,5],
        [8,9,7],
        [9,6,7],
        [9,5,6],
        [7,6,2]
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
    result_0 = -27.0 / 2.0 * helper_0 * y + 9 * helper_0 - 27.0 / 2.0 * helper_1 * x + 9 * helper_1 - 9.0 / 2.0 * pow(x, 3) + 18 * x * y - 11.0 / 2.0 * x - 9.0 / 2.0 * pow(y, 3) - 11.0 / 2.0 * y + 1

    return result_0

def lagr1(x,y):
    result_0 = (1.0 / 2.0) * x * (9 * pow(x, 2) - 9 * x + 2);
    return result_0

def lagr2(x,y):
    result_0 = (1.0 / 2.0) * y * (9 * pow(y, 2) - 9 * y + 2)
    return result_0

def lagr3(x,y):
    result_0 = (9.0 / 2.0) * x * (x + y - 1) * (3 * x + 3 * y - 2)
    return result_0

def lagr4(x,y):
    result_0 = -9.0 / 2.0 * x * (3 * pow(x, 2) + 3 * x * y - 4 * x - y + 1)
    return result_0

def lagr5(x,y):
    result_0 = (9.0 / 2.0) * x * y * (3 * x - 1)
    return result_0

def lagr6(x,y):
    result_0 = (9.0 / 2.0) * x * y * (3 * y - 1)
    return result_0

def lagr7(x,y):
    result_0 = -9.0 / 2.0 * y * (3 * x * y - x + 3 * pow(y, 2) - 4 * y + 1)
    return result_0

def lagr8(x,y):
    result_0 = (9.0 / 2.0) * y * (x + y - 1) * (3 * x + 3 * y - 2)
    return result_0

def lagr9(x,y):
    result_0 = -27 * x * y * (x + y - 1)
    return result_0

def eval_lagr(p, nodes):
	lagrs = [lagr0, lagr1, lagr2, lagr3, lagr4, lagr5, lagr6, lagr7, lagr8, lagr9]

	x = p[:,0]
	y = p[:,1]

	res = np.zeros((p.shape[0], nodes.shape[1]))
	
	for i,n in enumerate(nodes):
		res+=lagrs[i](x,y)[:, None]*n

	return res

def write_matlab_script(file):
    with open(file, "w") as f:
        f.write(
            """
C_filename = "matlab_input_C1_constraint_matrix.hdf5";
P_T_filename = "CT_P_T_matrix.hdf5";
points_filename = "matlab_mesh.hdf5";
Cone_filename = "matlab_input_cone_matrix.hdf5";
Di_filename = "matlab_input_dirichlet_matrix.hdf5";


% C
C_c = h5read(C_filename, "/C/cols");
C_r = h5read(C_filename, "/C/rows");
C_v = h5read(C_filename, "/C/values");
C_shape = h5read(C_filename, "/C/shape");
C = sparse(C_r+1,C_c+1,C_v,C_shape(1), C_shape(2));

% P_T
p_t_c = h5read(P_T_filename, "/P_T/cols");
p_t_r = h5read(P_T_filename, "/P_T/rows");
p_t_v = h5read(P_T_filename, "/P_T/values");
p_t_shape = h5read(P_T_filename, "/P_T/shape");
P_T = sparse(p_t_r+1, p_t_c+1, p_t_v, p_t_shape(1), p_t_shape(2));

% Dirichlet
Di_c = h5read(Di_filename, "/Di/cols");
Di_r = h5read(Di_filename, "/Di/rows");
Di_v = h5read(Di_filename, "/Di/values");
Di_shape = h5read(Di_filename, "/Di/shape");
Di = sparse(Di_r+1,Di_c+1,Di_v,Di_shape(1), Di_shape(2));

% Cone
Cone_c = h5read(Cone_filename, "/C_cone/cols");
Cone_r = h5read(Cone_filename, "/C_cone/rows");
Cone_v = h5read(Cone_filename, "/C_cone/values");
Cone_shape = h5read(Cone_filename, "/C_cone/shape");
Cone = sparse(Cone_r+1,Cone_c+1,Cone_v,Cone_shape(1), Cone_shape(2));

% triple C
C_c_trip = zeros(size(C_c, 1)*3, 1);
C_r_trip = zeros(size(C_r, 1)*3, 1);
C_v_trip = zeros(size(C_v, 1)*3, 1);

for i = 1:size(C_r, 1)
    C_r_trip((i - 1) * 3 + 1, 1) = (C_r(i,1)) * 3 + 0;
    C_r_trip((i - 1) * 3 + 2, 1) = (C_r(i,1)) * 3 + 1;
    C_r_trip((i - 1) * 3 + 3, 1) = (C_r(i,1)) * 3 + 2;
end

for i = 1:size(C_c, 1)
    C_c_trip((i - 1) * 3 + 1, 1) = (C_c(i,1)) * 3 + 0;
    C_c_trip((i - 1) * 3 + 2, 1) = (C_c(i,1)) * 3 + 1;
    C_c_trip((i - 1) * 3 + 3, 1) = (C_c(i,1)) * 3 + 2;
end

for i = 1:size(C_v, 1)
    C_v_trip((i - 1) * 3 + 1, 1) = C_v(i,1);
    C_v_trip((i - 1) * 3 + 2, 1) = C_v(i,1);
    C_v_trip((i - 1) * 3 + 3, 1) = C_v(i,1);
end

C_trip = sparse(C_r_trip+1,C_c_trip+1,C_v_trip, C_shape(1)*3, C_shape(2)*3);

% reorder Cone
Cone_ro_c = zeros(size(Cone_c, 1), 1);
Cone_ro_r = zeros(size(Cone_r, 1), 1);
Cone_ro_v = zeros(size(Cone_v, 1), 1);

Cone_ro_r = Cone_r;
Cone_ro_v = Cone_v;

for i = 1:size(Cone_c, 1)
    old_c = Cone_c(i);
    new_c = -1;
    if old_c < Cone_shape(2)/3
        new_c = old_c * 3;
    elseif old_c < Cone_shape(2)/3*2
        new_c = (old_c - Cone_shape(2)/3) * 3 + 1;
    else
        new_c = (old_c - Cone_shape(2)/3*2) * 3 + 2;
    end
    Cone_ro_c(i) = new_c;
end

Cone_ro_c = int64(Cone_ro_c);

Cone_ro = sparse(Cone_ro_r+1,Cone_ro_c+1,Cone_ro_v,Cone_shape(1), Cone_shape(2));


% triple Dirichlet
Di_c_trip = zeros(size(Di_c, 1)*3, 1);
Di_r_trip = zeros(size(Di_r, 1)*3, 1);
Di_v_trip = zeros(size(Di_v, 1)*3, 1);

for i = 1:size(Di_r, 1)
    Di_r_trip((i - 1) * 3 + 1, 1) = (Di_r(i,1)) * 3 + 0;
    Di_r_trip((i - 1) * 3 + 2, 1) = (Di_r(i,1)) * 3 + 1;
    Di_r_trip((i - 1) * 3 + 3, 1) = (Di_r(i,1)) * 3 + 2;
end

for i = 1:size(Di_c, 1)
    Di_c_trip((i - 1) * 3 + 1, 1) = (Di_c(i,1)) * 3 + 0;
    Di_c_trip((i - 1) * 3 + 2, 1) = (Di_c(i,1)) * 3 + 1;
    Di_c_trip((i - 1) * 3 + 3, 1) = (Di_c(i,1)) * 3 + 2;
end

for i = 1:size(Di_v, 1)
    Di_v_trip((i - 1) * 3 + 1, 1) = Di_v(i,1);
    Di_v_trip((i - 1) * 3 + 2, 1) = Di_v(i,1);
    Di_v_trip((i - 1) * 3 + 3, 1) = Di_v(i,1);
end

Di_trip = sparse(Di_r_trip+1,Di_c_trip+1,Di_v_trip, Di_shape(1)*3, Di_shape(2)*3);

Full_cons = [C_trip; Di_trip];
% Full_cons = [C_trip; Cone_ro; Di_trip];
% Full_cons = Di_trip;
Full_cons_permuted = Full_cons * P_T';

% points
points = h5read(points_filename, "/points");
points = points';
node_vector = reshape(points', 1, []);

b_cons = - C_trip * node_vector';
% b_cons = - [C_trip; Cone_ro] * node_vector';
b_di = zeros(size(Di_r_trip, 1), 1);
b = [b_cons; b_di];

% QR decomposition
% row_norm = 1 ./ max(abs(Full_cons_permuted), [], 2);
% row_norm = sparse(diag(row_norm));
% Full_cons_permuted = row_norm * Full_cons_permuted;
% b = row_norm * b;

% [Q, R_tmp, P_qr] = qr(Full_cons_permuted);
% 
% ind_rows = [];
% for i = 1:size(R_tmp, 1)
%     if sum(abs(R_tmp(i, :))) <= 1e-16
%         ind_rows = [ind_rows i];
%     end
% end
% 
% R = R_tmp(ind_rows, :);

[Q, R, P_qr] = qr(Full_cons_permuted);

R1 = R(1:end, 1:size(R,1));
R2 = R(1:end, size(R,1) + 1:end);

% useful things
P = P_qr' * P_T;
R1invR2 = R1\R2;
R1invQTb = R1 \ (Q' * b);
b_m = zeros(size(node_vector, 2), 1);
b_m(1:size(R1invQTb, 1)) = R1invQTb;
b_m = P' * b_m;
M = P' * [-R1invR2; speye(size(R1invR2, 2))];

[M_row, M_col, M_v] = find(M);
M_file_data = [size(M, 1) size(M, 2) size(M_row, 1);M_row M_col M_v];
writematrix(M_file_data, "matlab_M.txt");

[b_m_row, b_m_col, b_m_v] = find(b_m);
b_m_file_data = [size(b_m, 1) size(b_m, 2) size(b_m_row, 1);b_m_row b_m_col b_m_v];
writematrix(b_m_file_data, "matlab_b_m.txt");

[b_row, b_col, b_v] = find(b);
b_file_data = [size(b, 1) size(b, 2) size(b_row, 1);b_row b_col b_v];
writematrix(b_file_data, "matlab_b.txt");

[Full_cons_row, Full_cons_col, Full_cons_v] = find(Full_cons);
Full_cons_file_data = [size(Full_cons, 1) size(Full_cons, 2) size(Full_cons_row, 1);Full_cons_row Full_cons_col Full_cons_v];
writematrix(Full_cons_file_data, "matlab_C_trip.txt");

h5create("matlab_M.hdf5", "/A_proj_triplets/values", size(M_v, 1));
h5write("matlab_M.hdf5", "/A_proj_triplets/values", M_v');
h5create("matlab_M.hdf5", "/A_proj_triplets/rows", size(M_v, 1), Datatype="int32");
h5write("matlab_M.hdf5", "/A_proj_triplets/rows", int32(M_row - 1)');
h5create("matlab_M.hdf5", "/A_proj_triplets/cols", size(M_col, 1), Datatype="int32");
h5write("matlab_M.hdf5", "/A_proj_triplets/cols", int32(M_col - 1)');
h5create("matlab_M.hdf5", "/A_proj_triplets/shape", 2);
h5write("matlab_M.hdf5", "/A_proj_triplets/shape", [size(M, 1); size(M, 2)]');

h5create("matlab_M.hdf5", "/A_triplets/values", size(Full_cons_v, 1));
h5write("matlab_M.hdf5", "/A_triplets/values", Full_cons_v');
h5create("matlab_M.hdf5", "/A_triplets/rows", size(Full_cons_v, 1), Datatype="int32");
h5write("matlab_M.hdf5", "/A_triplets/rows", int32(Full_cons_row - 1)');
h5create("matlab_M.hdf5", "/A_triplets/cols", size(Full_cons_col, 1), Datatype="int32");
h5write("matlab_M.hdf5", "/A_triplets/cols", int32(Full_cons_col - 1)');
h5create("matlab_M.hdf5", "/A_triplets/shape", 2);
h5write("matlab_M.hdf5", "/A_triplets/shape", [size(Full_cons, 1); size(Full_cons, 2)]');

h5create("matlab_M.hdf5", "/b", [1,size(b, 1)]);
h5write("matlab_M.hdf5", "/b", b');
h5create("matlab_M.hdf5", "/b_proj", [1,size(b_m, 1)]);
h5write("matlab_M.hdf5", "/b_proj", b_m');

exit;
"""
        )


def read_matlab_sparse(filename):
    rows = []
    cols = []
    values = []
    r = -1
    c = -1
    size = -1
    with open(filename, "r") as f:
        row = 0
        for line in f:
            ss = line.rstrip("\n").split(",")
            # print(ss)
            assert len(ss) == 3
            if row == 0:
                r = int(ss[0])
                c = int(ss[1])
                size = int(ss[2])
            else:
                rows.append(int(ss[0]) - 1)
                cols.append(int(ss[1]) - 1)
                values.append(float(ss[2]))
            row += 1
    assert len(rows) == size
    assert r > 0 and c > 0
    rows = np.array(rows)
    cols = np.array(cols)
    values = np.array(values)
    spm = scipy.sparse.coo_array((values, (rows, cols)), shape=(r, c))

    return spm


# open JSON after validifying it
def is_valid_json(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    elif os.path.splitext(arg)[1] != ".json":
        parser.error("The file %s is not a .json file!" % arg)
    else:
        with open(arg, "r") as f:
            arg_json = json.load(f)
        return arg_json


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-j", dest="spec", required=True, type=lambda x: is_valid_json(parser, x)
    )
    parser.add_argument(
        "-b", dest="bins", required=True, type=lambda x: is_valid_json(parser, x)
    )

    args = parser.parse_args()

    input_file = args.spec[
        "input"
    ]  # vtu tetmesh file with 'winding_number' as cell data
    output_name = args.spec["output"]  # output name
    offset_file = args.spec["offset"]  # offset file
    weight_soft_1 = args.spec["weight_soft_1"]
    bilap_k_ring_neighbor = args.spec["bilap_k_ring_neighbor"] # int, bilaplacian on k ring 5 to 20
    bilap_sample_factor = args.spec["bilap_sample_factor"] # int, put 2
    elasticity_mode = args.spec["elasticity_mode"] # LinearElasticity or Neohookean
    enable_offset = args.spec["enable_offset"]

    path_to_para_exe = args.bins[
        "seamless_parametrization_binary"
    ]  # path to parametrization bin
    path_to_ct_exe = args.bins[
        "smooth_contours_binary"
    ]  # path to Clough Tocher constraints bin
    path_to_polyfem_exe = args.bins["polyfem_binary"]  # path to polyfem bin
    path_to_matlab_exe = args.bins["matlab_binary"]  # path to matlab exe
    path_to_toolkit_exe = args.bins["wmtk_c1_cone_split_binary"]  # path to toolkit app
    path_to_generate_cone_exe = args.bins["seamless_con_gen_binary"]


    # exit(0)

    # args = sys.argv

    # if len(args) < 8:
    #     print("Too few arguments. Expect 7")
    #     exit()
    # input_file = args[1]  # vtu tetmesh file with 'winding_number' as cell data
    # output_name = args[2]  # output name
    # path_to_para_exe = args[3]  # path to parametrization bin
    # path_to_ct_exe = args[4]  # path to Clough Tocher constraints bin
    # path_to_polyfem_exe = args[5]  # path to polyfem bin
    # path_to_matlab_exe = args[6]  # path to matlab exe
    # offset_file = args[7]  # offset file

    # workspace_path = args[4] # workspace path
    workspace_path = "./"

    tm = mio.read(input_file)
    vertices_unsliced = tm.points
    tets_unsliced = tm.cells_dict["tetra"]

    # check orientation, TODO: only do in debug
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

    # # use this code if want to keep the box
    tets = tets_unsliced
    vertices = vertices_unsliced
    winding_numbers = {}
    winding_numbers_data = winding_numbers_data_unsliced
    for i in range(tets.shape[0]):
        winding_numbers[i] = winding_numbers_data[i]
    # #and comment out the below

    # tet_indices_touching_surface = np.unique(np.argwhere(np.isin(tets_unsliced, surface_vertices))[:,0])

    # tets = tets_unsliced[tet_indices_touching_surface]
    # winding_numbers_data = winding_numbers_data_unsliced[tet_indices_touching_surface]
    # vertices = vertices_unsliced
    # vertices, tets, _, sliced_to_unsliced_v_map = igl.remove_unreferenced(vertices_unsliced, tets)

    # # print(winding_numbers_data.shape)
    # # print(tets.shape)
    # m_sliced = mio.Mesh(vertices, [('tetra', tets)], cell_data={"winding_number": winding_numbers_data.T})
    # m_sliced.write("test_slice.vtu")

    # # extract surface
    # winding_numbers = {}
    # for i in range(tets.shape[0]):
    #     winding_numbers[i] = winding_numbers_data[i]

    # # filtered_tets = np.array(filtered_tets)
    # # filtered_tets = []
    # # for i in range(tets.shape[0]):
    # #     if abs(winding_numbers_data[i]) >= 0.5:
    # #         filtered_tets.append(tets[i])
    # # filtered_tets = np.array(filtered_tets)

    # surface_tet_faces_unsliced = igl.boundary_facets(filtered_tets)
    # unsliced_to_sliced_v_map = {}
    # for i in range(len(sliced_to_unsliced_v_map)):
    #     unsliced_to_sliced_v_map[sliced_to_unsliced_v_map[i]] = i
    # surface_tet_faces = surface_tet_faces_unsliced.copy().tolist()
    # for i in range(len(surface_tet_faces)):
    #     for j in range(3):
    #         # print(surface_tet_face[i][j])
    #         surface_tet_faces[i][j] = unsliced_to_sliced_v_map[surface_tet_faces[i][j]]
    # surface_tet_faces = np.array(surface_tet_faces)
    # # print(surface_tet_faces)

    # get surface mesh for parametrization
    # surface_tet_faces_filtered = igl.boundary_facets(filtered_tets)

    para_in_v, para_in_f, im, para_in_v_to_tet_v_map = igl.remove_unreferenced(
        vertices, surface_tet_faces
    )
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
    # print(surface_in_tet_map)

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

    print(
        "[{}] ".format(datetime.datetime.now()),
        "computed tet->surface and surface->tet mapping.",
    )

    # do simplicial embedding
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

    # tets_regular = np.array(tets_regular)
    # tets_vertices_regular = np.array(tets_vertices_regular)

    print(
        "[{}] ".format(datetime.datetime.now()),
        "Done simplicial embedding. Splitted {} tets".format(simplicial_embedding_cnt),
    )

    ####################################################
    #             Call generate frame field            #
    ####################################################
    print("[{}] ".format(datetime.datetime.now()), "Calling generate frame field code")
    para_command = (
        path_to_generate_cone_exe
        + " --mesh "
        + workspace_path
        + "embedded_surface.obj --input ./"
    )
    # para_command = path_to_generate_cone_exe + " --mesh " + workspace_path + "tet.obj --input ./"

    subprocess.run(para_command, shell=True, check=True)

    # rearrange cones
    angles = np.loadtxt("embedded_surface_Th_hat")

    cone_vids = []
    for i, angle in enumerate(angles):
        if angle < 6.0 or angle > 6.5:  # less or more than 2 * pi
            cone_vids.append(i)
    print("cone cnt: ", len(cone_vids))
    v_embd, _, _, f_embd, _, _ = igl.read_obj("embedded_surface.obj")

    adjlist = igl.adjacency_list(f_embd)
    colored = [False] * v_embd.shape[0]
    cone_list = copy.deepcopy(cone_vids)

    cannot_place_id = []
    placed_id = []
    for c in cone_list:
        if not colored[c]:
            placed_id.append(c)
            colored[c] = True
            for adjv in adjlist[c]:
                colored[adjv] = True
        else:
            cannot_place_id.append(c)

    exchange_map = {}
    second_failed = []

    for c in cannot_place_id:
        # try place in one ring
        found = False
        for adjv in adjlist[c]:
            if not colored[adjv]:
                exchange_map[c] = adjv
                exchange_map[adjv] = c
                colored[adjv] = True
                for vv in adjlist[adjv]:
                    colored[vv] = True
                found = True
                break
        if not found:
            second_failed.append(c)

    third_failed = []
    for c in second_failed:
        found = False
        for vv in range(v_embd.shape[0]):
            if not colored[vv]:
                exchange_map[c] = vv
                exchange_map[vv] = c
                colored[vv] = True
                for vvv in adjlist[vv]:
                    colored[vvv] = True
                found = True
                break
        if not found:
            third_failed.append(c)

    if len(third_failed) != 0:
        print("cannot easily place cone! try make mesh denser")
        exit()

    reorder_vid = []
    for i in range(v_embd.shape[0]):
        if i not in exchange_map:
            reorder_vid.append(i)
        else:
            reorder_vid.append(exchange_map[i])

    assert len(reorder_vid) == v_embd.shape[0]

    with open("embedded_surface_Th_hat_reordered", "w") as file:
        for i in range(v_embd.shape[0]):
            file.write("{}\n".format(angles[reorder_vid[i]]))

    angles_new = np.loadtxt("embedded_surface_Th_hat_reordered")
    cone_vids_new = []
    for i, angle_new in enumerate(angles_new):
        if angle_new < 6.0 or angle_new > 6.5:  # less or more than 2 * pi
            cone_vids_new.append(i)

    assert len(cone_vids_new) == len(cone_vids)


    ####################################################
    #             Call Parametrization Code            #
    ####################################################
    print("[{}] ".format(datetime.datetime.now()), "Calling parametrization code")
    para_command = (
        path_to_para_exe
        + " --mesh "
        + workspace_path
        + "embedded_surface.obj --cones embedded_surface_Th_hat --field embedded_surface_kappa_hat"
    )
    # para_command = (
    #     path_to_para_exe
    #     + " --mesh "
    #     + workspace_path
    #     + "embedded_surface.obj --cones embedded_surface_Th_hat_reordered --field embedded_surface_kappa_hat"
    # )
    # para_command = (
    #     path_to_para_exe
    #     + " --mesh "
    #     + workspace_path
    #     + "embedded_surface.obj --cones cone_angles.txt --field embedded_surface_kappa_hat"
    # )

    subprocess.run(
        para_command,
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.DEVNULL,
    )

    ####################################################
    #       Split the tet with new para vertices       #
    ####################################################




    # do not comment out this !!!!
    tets_regular = np.array(tets_regular)
    tets_vertices_regular = np.array(tets_vertices_regular)

    ####################################################
    #             Call c1_meshing_split_app            #
    ####################################################
    para_out_file = (
        workspace_path + "parameterized_mesh.obj"
    )  # the file name para code should generate

    p_v, p_tc, _, p_f, p_ftc, _ = igl.read_obj(para_out_file)
    if p_v.shape[0] > para_in_v.shape[0]:
        print("do not support parametrization new vertex")
        exit()

    with open("toolkit_map.txt", "w") as file:
        # {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}
        for i in range(p_f.shape[0]):
            tid = surface_adj_tet[i][0]
            tet = tets_regular[tid]
            f = p_f[i]
            f_tet_base = np.array([para_in_v_to_tet_v_map[f[i]] for i in range(3)])
            tfs = np.array(
                [
                    [tet[1], tet[2], tet[3]],
                    [tet[0], tet[2], tet[3]],
                    [tet[0], tet[1], tet[3]],
                    [tet[0], tet[1], tet[2]],
                ]
            )
            found = False
            for k in range(4):
                if face_equal(f_tet_base, tfs[k]):
                    file.write("{} {}\n".format(tid, k))
                    found = True
            assert found

    angles = np.loadtxt("embedded_surface_Th_hat")
    # angles = np.loadtxt("embedded_surface_Th_hat_reordered")

    cone_vids = []
    for i, angle in enumerate(angles):
        if angle < 6.0 or angle > 6.5:  # less or more than 2 * pi
            cone_vids.append(i)
    print("cone cnt: ", len(cone_vids))

    with open("toolkit_cone_edges.txt", "w") as f:
        edges = igl.edges(p_f)
        for e in edges:
            if e[0] in cone_vids and e[1] in cone_vids:
                f.write("{} {}\n".format(e[0], e[1]))

        # vv_adjacency = igl.adjacency_list(p_f)
        # edges_to_split = []
        # for i in range(len(vv_adjacency)):
        #     if i in cone_vids:
        #         for j in vv_adjacency[i]:
        #             if i<j:
        #                 edges_to_split.append([i,j])
        #             else:
        #                 edges_to_split.append([j,i])
        # edges_to_split = np.array(edges_to_split)
        # unique_edges = np.unique(edges_to_split, axis=0)

        # for e in unique_edges:
        #     f.write("{} {}\n".format(e[0], e[1]))

    with open("toolkit_para_edges.txt", "w") as f:
        a = 1

    toolkit_tet_points = tets_vertices_regular
    toolkit_tet_cells = [("tetra", tets_regular)]
    toolkit_tet = mio.Mesh(toolkit_tet_points, toolkit_tet_cells)
    toolkit_tet.write("toolkit_tet.msh", file_format="gmsh")

    toolkit_surface_points = p_v
    toolkit_surface_cells = [("triangle", p_f)]
    toolkit_surface = mio.Mesh(toolkit_surface_points, toolkit_surface_cells)
    toolkit_surface.write("toolkit_surface.msh", file_format="gmsh")

    toolkit_uv_points = p_tc
    toolkit_uv_cells = [("triangle", p_ftc)]
    toolkit_uv = mio.Mesh(toolkit_uv_points, toolkit_uv_cells)
    toolkit_uv.write("toolkit_uv.msh", file_format="gmsh")

    toolkit_json = {
        "tetmesh": "toolkit_tet.msh",
        "surface_mesh": "toolkit_surface.msh",
        "uv_mesh": "toolkit_uv.msh",
        "tet_surface_map": "toolkit_map.txt",
        "parametrization_edges": "toolkit_para_edges.txt",
        "adjacent_cone_edges": "toolkit_cone_edges.txt",
        "output": "toolkit",
    }

    with open("cone_split_json.json", "w") as f:
        json.dump(toolkit_json, f)

    print("[{}] ".format(datetime.datetime.now()), "Calling toolkit c1 cone splitting")
    toolkit_command = (
        path_to_toolkit_exe + " -j " + workspace_path + "cone_split_json.json"
    )
    subprocess.run(toolkit_command, shell=True, check=True)

    # exit()

    ####################################################
    #          Split if two cones are adjacent         #
    ####################################################
    # para_out_file = workspace_path + "parameterized_mesh.obj" # the file name para code should generate
    # # para_out_file = workspace_path + "parameterized_tet.obj" # the file name para code should generate

    # p_v, p_tc, _, p_f, p_ftc, _ = igl.read_obj(para_out_file)

    # angles = np.loadtxt("embedded_surface_Th_hat")
    # # angles = np.loadtxt("tet_Th_hat")

    # cone_vids = []
    # for i, angle in enumerate(angles):
    #     if angle < 6.0 or angle > 6.5: # less or more than 2 * pi
    #         cone_vids.append(i)
    # print(cone_vids)

    # e2f_map = {}
    # v2tc_map = {}
    # for i in range(p_v.shape[0]):
    #     v2tc_map[i] = []

    # for i in range(p_f.shape[0]):
    #     f = p_f[i]

    #     # e to f map
    #     e01 = str(f[0])+"+"+str(f[1])
    #     e10 = str(f[1])+"+"+str(f[0])
    #     e12 = str(f[1])+"+"+str(f[2])
    #     e21 = str(f[2])+"+"+str(f[1])
    #     e20 = str(f[2])+"+"+str(f[0])
    #     e02 = str(f[0])+"+"+str(f[2])
    #     es = [e01,e10,e12,e21,e20,e02]
    #     for e in es:
    #         if e not in e2f_map:
    #             e2f_map[e] = [i]
    #         else:
    #             e2f_map[e].append(i)

    #     # v to tc map
    #     for k in range(3):
    #         # print("p_ftc[i][k]", p_ftc[i][k])
    #         # print("v2tc_map[f[k]]", v2tc_map[f[k]])
    #         if p_ftc[i][k] not in v2tc_map[f[k]]:
    #             v2tc_map[f[k]].append(p_ftc[i][k])

    # new_vs = copy.deepcopy(p_v.tolist())
    # new_fs = copy.deepcopy(p_f.tolist())
    # new_tc = copy.deepcopy(p_tc.tolist())
    # new_ftc = copy.deepcopy(p_ftc.tolist())

    # p_edges = igl.edges(p_f)

    # affected_fids = []

    # for i in range(p_edges.shape[0]):
    #     e = p_edges[i]
    #     if e[0] in cone_vids and e[1] in cone_vids:
    #         print("split edge with two cones {} {}".format(e[0], e[1]))
    #         # edge connecting two cones, need to split
    #         v_new = (p_v[e[0]] + p_v[e[1]]) /2.0
    #         v_new_id = len(new_vs)
    #         new_vs.append(v_new)

    #         e2f_map[str(e[1]) + "+" + str(v_new_id)] = []
    #         e2f_map[str(e[0]) + "+" + str(v_new_id)] = []
    #         e2f_map[str(v_new_id) + "+" + str(e[0])] = []
    #         e2f_map[str(v_new_id) + "+" + str(e[1])] = []

    #         e_str = str(e[0]) + "+" + str(e[1])

    #         # split uv
    #         # consider different mapping for the two faces
    #         last_pair = [-1,-1]
    #         new_uvs = []
    #         new_uv_vids = []

    #         assert len(e2f_map[e_str])<=2

    #         affected_fids.extend(e2f_map[e_str])
    #         for fid in e2f_map[e_str]:
    #             tc_v0 = -1
    #             tc_v1 = -1
    #             # print(new_fs[fid])
    #             for k in range(3):
    #                 if new_fs[fid][k] == e[0]:
    #                     tc_v0 = new_ftc[fid][k]
    #                 if new_fs[fid][k] == e[1]:
    #                     tc_v1 = new_ftc[fid][k]
    #             assert tc_v0 > -1
    #             assert tc_v1 > -1
    #             new_uv = None
    #             new_uv_vid = -1
    #             if (tc_v0 in last_pair) and (tc_v1 in last_pair):
    #                 # use last
    #                 new_uv = new_uvs[0]
    #                 new_uv_vid = new_uv_vids[0]
    #             else:
    #                 # use new
    #                 new_uv = (p_tc[tc_v0] + p_tc[tc_v1])/2.0
    #                 new_uv_vid = len(new_tc)
    #                 new_tc.append(new_uv)
    #                 last_pair = [tc_v0, tc_v1]
    #             assert new_uv is not None
    #             assert new_uv_vid > -1

    #             new_uvs.append(new_uv)
    #             new_uv_vids.append(new_uv_vid)

    #             ftc_new_2 = copy.deepcopy(new_ftc[fid])
    #             # ftc_new_id = len(new_fs)

    #             # in place change
    #             for k in range(3):
    #                 if new_ftc[fid][k] == tc_v0:
    #                     new_ftc[fid][k] = new_uv_vid

    #             # change in ftc2
    #             for k in range(3):
    #                 if ftc_new_2[k] == tc_v1:
    #                     ftc_new_2[k] = new_uv_vid

    #             new_ftc.append(ftc_new_2)

    #         # split 3d
    #         for fid in e2f_map[e_str]:
    #             f_new_2 = copy.deepcopy(new_fs[fid])
    #             f_new_id = len(new_fs)

    #             other_vid = -1
    #             for k in range(3):
    #                 if new_fs[fid][k] != e[0] and new_fs[fid][k] != e[1]:
    #                     other_vid = new_fs[fid][k]
    #             assert other_vid>-1

    #             #in place change for f_new_1 for e[0]
    #             found = False
    #             for k in range(3):
    #                 if new_fs[fid][k] == e[0]:
    #                     found = True
    #                     new_fs[fid][k] = v_new_id
    #             assert found
    #             e2f_map[str(e[1]) + "+" + str(v_new_id)].append(fid)
    #             e2f_map[str(v_new_id) + "+" + str(e[1])].append(fid)
    #             # change in f_new_2 for e[1]
    #             found = False
    #             for k in range(3):
    #                 if f_new_2[k] == e[1]:
    #                     found = True
    #                     f_new_2[k] = v_new_id
    #             assert found
    #             e2f_map[str(e[0]) + "+" + str(v_new_id)].append(f_new_id)
    #             e2f_map[str(v_new_id) + "+" + str(e[0])].append(f_new_id)

    #             # update map for edge e0e2
    #             e2f_map[str(e[0]) + "+" + str(other_vid)].append(f_new_id)
    #             e2f_map[str(other_vid) + "+" + str(e[0])].append(f_new_id)
    #             e2f_map[str(e[0]) + "+" + str(other_vid)].remove(fid)
    #             e2f_map[str(other_vid) + "+" + str(e[0])].remove(fid)

    #             # append f_new_2
    #             new_fs.append(f_new_2)

    #             print(fid, " --> ", [fid, f_new_id])

    # new_vs = np.array(new_vs)
    # new_fs = np.array(new_fs)
    # new_tc = np.array(new_tc)
    # new_ftc = np.array(new_ftc)

    # assert new_fs.shape[0] == new_ftc.shape[0]

    # with open("parameterized_mesh_splitted.obj", 'w') as f:
    #     for i in range(new_vs.shape[0]):
    #         f.write("v {} {} {}\n".format(new_vs[i][0], new_vs[i][1], new_vs[i][2]))
    #     for i in range(new_tc.shape[0]):
    #         f.write("vt {} {}\n".format(new_tc[i][0], new_tc[i][1]))
    #     for i in range(new_fs.shape[0]):
    #         f.write("f {}/{} {}/{} {}/{}\n".format(new_fs[i][0]+1, new_ftc[i][0]+1, new_fs[i][1]+1, new_ftc[i][1]+1, new_fs[i][2]+1, new_ftc[i][2]+1))

    # print("affected fids: ", affected_fids)

    # # exit()

    # bd_f = igl.boundary_facets(tets_regular)
    # bd_v = []
    # for ff in bd_f:
    #     # corners
    #     for i in range(3):
    #         bd_v.append(ff[i])
    # bd_v = np.unique(np.array(bd_v))

    # bd_v_tag = np.array([int(i in bd_v) for i in range(tets_vertices_regular.shape[0])])
    # sf_v_tag = np.array([int(i in para_in_v_to_tet_v_map) for i in range(tets_vertices_regular.shape[0])])
    # both_v_tag = np.array([int(i in bd_v and i in para_in_v_to_tet_v_map) for i in range(tets_vertices_regular.shape[0])])

    # print("both before para split: ", np.sum(both_v_tag))

    # m_test_nodes_points = tets_vertices_regular
    # m_test_nodes_cells = [('tetra', tets_regular)]
    # m_test_nodes = mio.Mesh(m_test_nodes_points, m_test_nodes_cells)
    # m_test_nodes.point_data['boundary'] = bd_v_tag
    # m_test_nodes.point_data['surface_v'] = sf_v_tag
    # m_test_nodes.point_data['both'] = both_v_tag
    # # m.write('test_boundary_nodes.msh', file_format='gmsh')
    # m_test_nodes.write('test_before_para_split.vtu')

    ####################################################
    #                     Para Split                   #
    ####################################################
    # # get para in and out mapping
    # print("[{}] ".format(datetime.datetime.now()), "Doing parametrization input to output mapping ...")
    # # para_out_file = workspace_path + "parameterized_mesh.obj" # the file name para code should generate
    # para_out_file = workspace_path + "parameterized_mesh_splitted.obj"
    # para_out_v, para_out_tc, _, para_out_f, para_out_ftc, _ = igl.read_obj(para_out_file)

    # print("[{}] ".format(datetime.datetime.now()), "after para #v: {0}, before para #v: {1}".format(para_out_v.shape[0], para_in_v.shape[0]))
    # v_thres = para_in_v.shape[0]

    # # old face existance
    # new_face_ids = []
    # for i in range(para_out_f.shape[0]):
    #     if any(v_out >= v_thres for v_out in para_out_f[i]):
    #         new_face_ids.append(i)

    # deleted_old_fids = []
    # used_new_fids = [False for f in para_out_f]

    # # face dict existance
    # para_in_face_existance = {}
    # para_out_face_existance = {}
    # # for i in range(para_in_f.shape[0]):
    # #     face = [para_in_f[i][0], para_in_f[i][1], para_in_f[i][2]]
    # #     face.sort()
    # #     face_str = str(face[0]) + "+" + str(face[1]) + "+" + str(face[2])
    # #     para_in_face_existance[face_str] = i

    # for i in range(para_out_f.shape[0]):
    #     face = [para_out_f[i][0], para_out_f[i][1], para_out_f[i][2]]
    #     face.sort()
    #     face_str = str(face[0]) + "+" + str(face[1]) + "+" + str(face[2])
    #     para_out_face_existance[face_str] = i

    # para_in_to_out_face_mapping = {}

    # for i in range(para_in_f.shape[0]):
    #     face = [para_in_f[i][0], para_in_f[i][1], para_in_f[i][2]]
    #     face.sort()
    #     face_str = str(face[0]) + "+" + str(face[1]) + "+" + str(face[2])
    #     if face_str in para_out_face_existance:
    #         para_in_to_out_face_mapping[i] = [para_out_face_existance[face_str]]
    #     else:
    #         deleted_old_fids.append(i)

    # # for i in range(para_in_f.shape[0]):
    # #     found = False
    # #     for j in range(para_out_f.shape[0]):
    # #         if used_new_fids[j]:
    # #             continue
    # #         if face_equal(para_in_f[i], para_out_f[j]):
    # #             found = True
    # #             used_new_fids[j] = True
    # #             para_in_to_out_face_mapping[i] = [j]
    # #     if not found:
    # #         deleted_old_fids.append(i)

    # print("[{}] ".format(datetime.datetime.now()), "Done parametrization mapping.")

    # # match new faces to old faces
    # print("[{}] ".format(datetime.datetime.now()), "Compute old -> new face containment")
    # for f_out in new_face_ids:
    #     bc = (para_out_v[para_out_f[f_out][0]] + para_out_v[para_out_f[f_out][1]] + para_out_v[para_out_f[f_out][2]]) / 3.0
    #     found = False
    #     for f_in in deleted_old_fids:
    #         if on_tri(para_in_v[para_in_f[f_in][0]], para_in_v[para_in_f[f_in][1]], para_in_v[para_in_f[f_in][2]], bc):
    #             if f_in in para_in_to_out_face_mapping:
    #                 para_in_to_out_face_mapping[f_in].append(f_out)
    #             else:
    #                 para_in_to_out_face_mapping[f_in] = [f_out]
    #             found = True
    #             break
    #     assert found # must find an f_in that contains f_out

    # print("deleted_old_fids: ", deleted_old_fids)
    # for fid in deleted_old_fids:
    #     print(fid, ": ", para_in_to_out_face_mapping[fid])

    # tet_after_para_vertices = copy.deepcopy(tets_vertices_regular.tolist())
    # tet_after_para_tets = copy.deepcopy(tets_regular.tolist())

    # # add new vertices to tet mesh
    # print("[{}] ".format(datetime.datetime.now()), "Adding parametrized new vertices to tetmesh ... ")
    # para_out_v_to_tet_v_map = copy.deepcopy(para_in_v_to_tet_v_map).tolist()
    # print(para_in_v.shape)
    # print(para_out_v.shape)
    # for i in range(para_in_v.shape[0], para_out_v.shape[0]):
    #     # para_out_v_to_tet_v_map[i] = len(tet_after_para_vertices)
    #     para_out_v_to_tet_v_map.append(len(tet_after_para_vertices))
    #     tet_after_para_vertices.append(para_out_v[i].tolist())

    # # exit()
    # assert len(para_out_v_to_tet_v_map) == para_out_v.shape[0]

    # # para_out to tet_out surface mappings
    # surface_adj_tet_para_out = {}
    # tet_surface_para_out = {}

    # check_cnt = 0
    # for key in para_in_to_out_face_mapping:
    #     check_cnt += len(para_in_to_out_face_mapping[key])
    # assert check_cnt == para_out_f.shape[0]

    # print("[{}] ".format(datetime.datetime.now()), "Splitting tets according to para output ... ")
    # # update unsplitted faces
    # for f_in in para_in_to_out_face_mapping:
    #     if f_in not in deleted_old_fids:
    #         assert len(para_in_to_out_face_mapping[f_in]) == 1
    #         surface_adj_tet_para_out[para_in_to_out_face_mapping[f_in][0]] = surface_adj_tet[f_in]
    #         for tid in surface_adj_tet_para_out[para_in_to_out_face_mapping[f_in][0]]:
    #             tet_surface_para_out[tid] = [para_in_to_out_face_mapping[f_in][0]]

    # # split corresponding tets
    # print("para deleted fids: ", deleted_old_fids)
    # for f_in in deleted_old_fids:
    #     f_vs = surface_tet_faces[f_in] # vid in tet regular index
    #     print("-------------------------------------")
    #     print("old fid: ", f_in, "  f_vs: ", f_vs)
    #     adj_tets = surface_adj_tet[f_in]
    #     assert len(adj_tets) == 2
    #     for t in adj_tets:
    #         t_vs = tets_regular[t]

    #         # get local ids for f_vs and other point
    #         local_ids = [-1,-1,-1,-1]
    #         for i in range(3):
    #             for j in range(4):
    #                 if f_vs[i] == t_vs[j]:
    #                     local_ids[j] = i
    #                     break
    #         assert local_ids.count(-1) == 1

    #         new_tets = []
    #         assert len(para_in_to_out_face_mapping[f_in]) > 1
    #         for f_out in para_in_to_out_face_mapping[f_in]:
    #             f_out_vs = para_out_f[f_out]
    #             f_out_vs_tet_base = [para_out_v_to_tet_v_map[vid] for vid in f_out_vs]
    #             print("f_out_vs_tet_base: ", f_out_vs_tet_base)
    #             new_tet_vs = copy.deepcopy(t_vs)
    #             for i in range(4):
    #                 if local_ids[i] != -1:
    #                     new_tet_vs[i] = f_out_vs_tet_base[local_ids[i]]
    #             new_tets.append(new_tet_vs)

    #         print("old tet: ", t_vs)
    #         print("new_tets: ", new_tets)

    #         # propagate winding number
    #         old_winding_number = winding_numbers[t]

    #         # update tet connectivity
    #         assert len(new_tets) > 1
    #         tet_after_para_tets[t] = new_tets[0]
    #         first_split_face = para_in_to_out_face_mapping[f_in][0]
    #         if first_split_face not in surface_adj_tet_para_out:
    #             surface_adj_tet_para_out[first_split_face] = [t]
    #         else:
    #             surface_adj_tet_para_out[first_split_face].append(t)
    #         tet_surface_para_out[t] = [first_split_face]

    #         # propagate winding number
    #         winding_numbers[t] = winding_numbers[t]

    #         for i in range(1, len(new_tets)):
    #             new_tid = len(tet_after_para_tets)
    #             # print(new_tid)
    #             tet_after_para_tets.append(new_tets[i])
    #             split_face = para_in_to_out_face_mapping[f_in][i]
    #             if split_face not in surface_adj_tet_para_out:
    #                 surface_adj_tet_para_out[split_face] = [new_tid]
    #             else:
    #                 surface_adj_tet_para_out[split_face].append(new_tid)
    #             tet_surface_para_out[new_tid] = [split_face]

    #             # propagate winding number
    #             winding_numbers[new_tid] = old_winding_number

    #     # figure out how edges are splitted

    # # split tets surrounding the splitted

    # print("[{}] ".format(datetime.datetime.now()), "Done Para Split.")

    # bd_f = igl.boundary_facets(np.array(tet_after_para_tets))
    # bd_v = []
    # for ff in bd_f:
    #     # corners
    #     for i in range(3):
    #         bd_v.append(ff[i])
    #     if any(ff[k] in para_out_v_to_tet_v_map for k in range(3)):
    #         print("has vertex on boundary: ", ff)
    # bd_v = np.unique(np.array(bd_v))

    # bd_v_tag = np.array([int(i in bd_v) for i in range(len(tet_after_para_vertices))])
    # sf_v_tag = np.array([int(i in para_out_v_to_tet_v_map) for i in range(len(tet_after_para_vertices))])
    # both_v_tag = np.array([int(i in bd_v and i in para_out_v_to_tet_v_map) for i in range(len(tet_after_para_vertices))])

    # print("both after para split: ", np.sum(both_v_tag))

    # m_test_nodes_points = np.array(tet_after_para_vertices)
    # m_test_nodes_cells = [('tetra', np.array(tet_after_para_tets))]
    # m_test_nodes = mio.Mesh(m_test_nodes_points, m_test_nodes_cells)
    # m_test_nodes.point_data['boundary'] = bd_v_tag
    # m_test_nodes.point_data['surface_v'] = sf_v_tag
    # m_test_nodes.point_data['both'] = both_v_tag
    # # m.write('test_boundary_nodes.msh', file_format='gmsh')
    # m_test_nodes.write('test_after_para_split.vtu')

    ####################################################
    #                     Face Split                   #
    ####################################################
    tet_after_para_mesh = mio.read("toolkit_tetmesh_tets.vtu")
    tet_after_para_vertices = tet_after_para_mesh.points.tolist()
    # tet_after_para_tets = tet_after_para_mesh.cells_dict['tetra'].tolist()

    tet_after_para_tets = tet_after_para_mesh.cells_dict["tetra"]
    tet_after_para_tets = tet_after_para_tets[:, [1, 0, 2, 3]]
    tet_after_para_tets = tet_after_para_tets.tolist()

    para_out_v, para_out_tc, _, para_out_f, para_out_ftc, _ = igl.read_obj(
        "surface_uv_after_cone_split.obj"
    )

    para_out_v_to_tet_v_map = np.loadtxt(
        "surface_v_to_tet_v_after_cone_split.txt"
    ).astype(np.int64)

    print("para_out_v_to_tet_v_map.shape", para_out_v_to_tet_v_map.shape)

    surface_adj_tet_para_out = {}
    tet_surface_para_out = {}
    surface_adj_tet_from_file = np.loadtxt(
        "surface_adj_tet_after_cone_split.txt"
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

    ####################################################
    #                     Call Gmsh                    #
    ####################################################
    print("[{}] ".format(datetime.datetime.now()), "Calling Gmsh ... ")
    gmsh.initialize()
    gmsh.open(workspace_path + "tetmesh_after_face_split.msh")
    gmsh.model.mesh.setOrder(3)
    gmsh.write(workspace_path + "tetmesh_after_face_split_high_order_tet.msh")
    gmsh.write(workspace_path + "tetmesh_after_face_split_high_order_tet.m")

    ####################################################
    #                   Call CT Code                   #
    ####################################################

    print("[{}] ".format(datetime.datetime.now()), "Calling Clough Tocher code")
    # ct_command = path_to_ct_exe + " --input " + workspace_path + "parameterized_mesh_splitted.obj -o CT"
    ct_command = (
        path_to_ct_exe
        + " --input "
        + workspace_path
        + "surface_uv_after_cone_split.obj -o CT"
    )

    subprocess.run(ct_command, shell=True, check=True)
    # subprocess.run(ct_command.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    ####################################################
    #                   Smooth cones                   #
    ####################################################
    print("[{}] ".format(datetime.datetime.now()), "smoothing cone area")

    c_area_vertices = np.loadtxt("CT_bilaplacian_nodes_values_cone_area_vertices.txt").astype(np.int32)
    c_area_vertices = np.unique(c_area_vertices)

    ryan_mesh = mio.read("CT_from_lagrange_nodes.msh")
    cone_area_face = np.loadtxt("CT_bilaplacian_nodes_values_cone_area_faces.txt").astype(np.int32)
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
    # v_smoothed = scipy.sparse.linalg.spsolve(scipy.sparse.identity(L_w.shape[0]) - 1 * L_w, v_ct)
    # v_smoothed = (True, v_smoothed)

    igl.write_obj("sampled_after_smoothing.obj", v_smoothed[1], f_ct)

    # v_hack, _, _, f_hack, _, _ = igl.read_obj("sampled_after_smoothing_new.obj")
    # v_smoothed = (True, v_hack)

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


    # exit()

    ####################################################
    #             Call CT with new normals             #
    ####################################################

    print("[{}] ".format(datetime.datetime.now()), "Calling Clough Tocher code with new normals")
    ct_command_2 = (
        path_to_ct_exe
        + " --input "
        + workspace_path
        + "surface_uv_after_cone_split.obj -o CT --vertex_normals CT_smoothed_normals.txt"
    )

    subprocess.run(ct_command_2, shell=True, check=True)


    ####################################################
    #          Map tri langrange nodes to tet          #
    ####################################################

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

    ####################################################
    #           Constuct Constraint Matrix             #
    ####################################################
    print(
        "[{}] ".format(datetime.datetime.now()),
        "constructing full hard constraint matrix ...",
    )

    interior_matrix = scipy.io.mmread("CT_interior_constraint_matrix.txt")
    edge_end_point_matrix = scipy.io.mmread(
        "CT_edge_endpoint_constraint_matrix_eliminated.txt"
    )
    edge_mid_point_matrix = scipy.io.mmread("CT_edge_midpoint_constraint_matrix.txt")

    full_matrix = scipy.sparse.vstack(
        (interior_matrix, edge_end_point_matrix, edge_mid_point_matrix)
    )

    local2global = np.loadtxt(
        workspace_path + tri_to_tet_index_mapping_file_name
    ).astype(np.int32)
    A = full_matrix.tocoo(True)
    m = mio.read(workspace_path + linear_tet_file_name)
    # write points hdf5 for matlab
    with h5py.File(workspace_path + "matlab_mesh.hdf5", "w") as f:
        f.create_dataset("points", data=m.points)

    v = m.points
    b = -(A @ v[local2global, :])

    with h5py.File(workspace_path + "CT_full_constraint_matrix.hdf5", "w") as f:
        f.create_dataset("local2global", data=local2global.astype(np.int32))
        f.create_dataset("A_triplets/values", data=A.data)
        f.create_dataset("A_triplets/cols", data=A.col)
        f.create_dataset("A_triplets/rows", data=A.row)
        f.create_dataset("b", data=b)

    local2global_matrix_rows = [i for i in range(local2global.shape[0])]
    local2global_matrix_cols = [local2global[i] for i in range(local2global.shape[0])]
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
        f["weight_triplets"].attrs["shape"] = (local2global.shape[0], v.shape[0])

    ####################################################
    #           Expanding Constraint Matrix            #
    ####################################################
    interior = scipy.io.mmread("CT_interior_constraint_matrix.txt")
    midpoint = scipy.io.mmread("CT_edge_midpoint_constraint_matrix.txt")
    endpoint = scipy.io.mmread("CT_edge_endpoint_constraint_matrix_eliminated.txt")
    cone = scipy.io.mmread("CT_cone_constraint_matrix.txt")

    local2global = np.loadtxt(
        workspace_path + tri_to_tet_index_mapping_file_name
    ).astype(np.int32)

    stacked = scipy.sparse.vstack((interior, endpoint, midpoint))

    stacked = stacked.tocoo(True)
    stacked_row = stacked.row
    stacked_col = stacked.col
    stacked_data = stacked.data

    stacked_tet_row = stacked_row
    stacked_tet_col = np.array(
        [local2global[stacked_col[i]] for i in range(stacked_col.shape[0])]
    )
    stacked_tet_data = stacked_data

    stacked_tet = scipy.sparse.coo_array(
        (stacked_tet_data, (stacked_tet_row, stacked_tet_col)),
        shape=(stacked.shape[0], v.shape[0]),
    )

    # stacked_tet_row = stacked_row
    # stacked_tet_col = stacked_col
    # stacked_tet_data = stacked_data

    # stacked_tet = stacked

    stacked_trip_row = [0 for i in range(stacked_tet_row.shape[0] * 3)]
    stacked_trip_col = [0 for i in range(stacked_tet_col.shape[0] * 3)]
    stacked_trip_data = [0 for i in range(stacked_tet_data.shape[0] * 3)]

    for i in range(stacked_tet_row.shape[0]):
        stacked_trip_row[i * 3 + 0] = stacked_tet_row[i] * 3 + 0
        stacked_trip_row[i * 3 + 1] = stacked_tet_row[i] * 3 + 1
        stacked_trip_row[i * 3 + 2] = stacked_tet_row[i] * 3 + 2

        stacked_trip_col[i * 3 + 0] = stacked_tet_col[i] * 3 + 0
        stacked_trip_col[i * 3 + 1] = stacked_tet_col[i] * 3 + 1
        stacked_trip_col[i * 3 + 2] = stacked_tet_col[i] * 3 + 2

        stacked_trip_data[i * 3 + 0] = stacked_tet_data[i]
        stacked_trip_data[i * 3 + 1] = stacked_tet_data[i]
        stacked_trip_data[i * 3 + 2] = stacked_tet_data[i]

    stacked_trip = scipy.sparse.coo_array(
        (stacked_trip_data, (stacked_trip_row, stacked_trip_col)),
        shape=(stacked_tet.shape[0] * 3, stacked_tet.shape[1] * 3),
    )

    cone = cone.tocoo(True)
    cone_row = cone.row
    cone_col = cone.col
    cone_data = cone.data
    local2global_cone = np.array(
        local2global.tolist()
        + (local2global + v.shape[0]).tolist()
        + (local2global + v.shape[0] * 2).tolist()
    )

    cone_tet_row = cone_row
    cone_tet_col = np.array(
        [local2global_cone[cone_col[i]] for i in range(cone_col.shape[0])]
    )
    cone_tet_data = cone_data
    cone_tet = scipy.sparse.coo_array(
        (cone_tet_data, (cone_tet_row, cone_tet_col)),
        shape=(cone.shape[0], v.shape[0] * 3),
    )

    # cone_tet_row = cone_row
    # cone_tet_col = cone_col
    # cone_tet_data = cone_data
    # cone_tet = cone

    cone_trip_row = [0 for i in range(cone_tet_row.shape[0])]
    cone_trip_col = [0 for i in range(cone_tet_col.shape[0])]
    cone_trip_data = [0 for i in range(cone_tet_data.shape[0])]

    cone_trip_row = cone_tet_row
    cone_trip_data = cone_tet_data

    for i in range(cone_tet_col.shape[0]):
        old_c = cone_tet_col[i]
        new_c = -1
        if old_c < cone_tet.shape[1] / 3:
            new_c = old_c * 3
        elif old_c < cone_tet.shape[1] / 3 * 2:
            new_c = (old_c - cone_tet.shape[1] / 3) * 3 + 1
        else:
            new_c = (old_c - cone_tet.shape[1] / 3 * 2) * 3 + 2
        cone_trip_col[i] = new_c

    cone_trip = scipy.sparse.coo_array(
        (cone_trip_data, (cone_trip_row, cone_trip_col)),
        shape=(cone_tet.shape[0], cone_tet.shape[1]),
    )

    print("cone constraints rows: ", cone_trip.shape[0])
    full_trip = scipy.sparse.vstack((stacked_trip, cone_trip))

    # # debug try
    # full_trip = stacked_trip.copy()
    print("full trip rows: ", full_trip.shape[0])
    # print("full trip rank: ", np.linalg.matrix_rank(full_trip.todense()))

    # exit()

    v_flatten = v.flatten()
    b = -full_trip @ v_flatten

    with h5py.File("CT_constraint_with_cone_tet_ids.hdf5", "w") as f:
        f.create_dataset("A_triplets/values", data=full_trip.data)
        f.create_dataset("A_triplets/cols", data=full_trip.col.astype(np.int32))
        f.create_dataset("A_triplets/rows", data=full_trip.row.astype(np.int32))
        f.create_dataset("A_triplets/shape", data=full_trip.shape)
        f.create_dataset("b", data=b[:, None])

    ####################################################
    #                   Some old code                  #
    ####################################################

    # compute boundary nodes
    cells_ho = m.cells_dict["tetra20"]
    cells = cells_ho[:, :4]
    bd_f = igl.boundary_facets(cells)
    bd_v = []
    for ff in bd_f:
        # corners
        for i in range(3):
            bd_v.append(ff[i])
        # edges
        e01 = str(ff[0]) + "+" + str(ff[1])
        e12 = str(ff[1]) + "+" + str(ff[2])
        e20 = str(ff[2]) + "+" + str(ff[0])
        bd_v.extend(tet_edge_to_vertices[e01])
        bd_v.extend(tet_edge_to_vertices[e12])
        bd_v.extend(tet_edge_to_vertices[e20])
        # faces
        ff_sorted = [ff[0], ff[1], ff[2]]
        ff_sorted.sort()
        ff_str = str(ff_sorted[0]) + "+" + str(ff_sorted[1]) + "+" + str(ff_sorted[2])
        bd_v.append(tet_face_to_vertices[ff_str])
    bd_v = np.unique(np.array(bd_v))

    bd_v_tag = np.array([int(i in bd_v) for i in range(v.shape[0])])
    sf_v_tag = np.array([int(i in local2global) for i in range(v.shape[0])])
    both_v_tag = np.array(
        [int(i in bd_v and i in local2global) for i in range(v.shape[0])]
    )

    print(bd_v_tag)

    m_test_nodes_points = m.points

    m_test_nodes_cells = [("tetra", cells)]
    m_test_nodes = mio.Mesh(m_test_nodes_points, m_test_nodes_cells)
    m_test_nodes.point_data["boundary"] = bd_v_tag
    m_test_nodes.point_data["surface_v"] = sf_v_tag
    m_test_nodes.point_data["both"] = both_v_tag
    # m.write('test_boundary_nodes.msh', file_format='gmsh')
    m_test_nodes.write("test_boundary_nodes.vtu")

    # bd_v = np.unique(bd_f)

    # build P_T, left mul x  i.e. P_T x
    # P right mul C i.e. C P
    l2g_r = np.flip(local2global)

    constrained = [False for i in range(v.shape[0])]
    for i in range(l2g_r.shape[0]):
        constrained[l2g_r[i]] = True
    appear_in_two = [False for i in range(v.shape[0])]
    for i in range(bd_v.shape[0]):
        if constrained[bd_v[i]] == True:
            print(bd_v[i], " is both boundary node and surface node!!")
            appear_in_two[bd_v[i]] = True
        constrained[bd_v[i]] = True
    unconstrained_v = []
    for i in range(len(constrained)):
        if not constrained[i]:
            unconstrained_v.append(i)

    print("tet v: ", v.shape[0])
    print("c1 cons v: ", l2g_r.shape[0])
    print("dirichlet cons v: ", bd_v.shape[0])
    print("unconstrained v: ", len(unconstrained_v))

    P_T = [-1 for i in range(v.shape[0])]
    cur_idx = 0
    for i in range(l2g_r.shape[0]):
        P_T[cur_idx] = l2g_r[i]
        cur_idx += 1
    for i in range(bd_v.shape[0]):
        if not (appear_in_two[bd_v[i]]):
            P_T[cur_idx] = bd_v[i]
            cur_idx += 1
    for i in range(len(unconstrained_v)):
        P_T[cur_idx] = unconstrained_v[i]
        cur_idx += 1
    assert cur_idx == v.shape[0]

    assert all(P_T[i] > -1 for i in range(len(P_T)))

    P_T_trip_row = []
    P_T_trip_col = []
    P_T_trip_value = [1.0 for i in range(len(P_T) * 3)]

    for i in range(len(P_T)):
        P_T_trip_row.append(i * 3 + 0)
        P_T_trip_row.append(i * 3 + 1)
        P_T_trip_row.append(i * 3 + 2)
    for i in range(len(P_T)):
        P_T_trip_col.append(P_T[i] * 3 + 0)
        P_T_trip_col.append(P_T[i] * 3 + 1)
        P_T_trip_col.append(P_T[i] * 3 + 2)

    with h5py.File(workspace_path + "CT_P_T_matrix.hdf5", "w") as f:
        f.create_dataset("P_T/values", data=np.array(P_T_trip_value))
        f.create_dataset("P_T/cols", data=np.array(P_T_trip_col))
        f.create_dataset("P_T/rows", data=np.array(P_T_trip_row))
        f.create_dataset(
            "P_T/shape", data=np.array([len(P_T_trip_value), len(P_T_trip_value)])
        )

    # add direchlet constraints
    di_row = [i for i in range(bd_v.shape[0])]
    di_col = [bd_v[i] for i in range(bd_v.shape[0])]
    di_value = [1.0 for i in range(bd_v.shape[0])]

    di_matrix = scipy.sparse.coo_array(
        (di_value, (di_row, di_col)), shape=(bd_v.shape[0], v.shape[0])
    )

    # expand constraint on tetmesh
    full_matrix_coo = full_matrix.tocoo(True)
    full_matrix_row = full_matrix_coo.row
    full_matrix_col = full_matrix_coo.col
    full_matrix_data = full_matrix_coo.data

    full_matrix_tet_row = full_matrix_row
    full_matrix_tet_col = np.array(
        [local2global[full_matrix_col[i]] for i in range(full_matrix_col.shape[0])]
    )
    full_matrix_tet_data = full_matrix_data

    full_matrix_tet = scipy.sparse.coo_array(
        (full_matrix_tet_data, (full_matrix_tet_row, full_matrix_tet_col)),
        shape=(full_matrix.shape[0], v.shape[0]),
    )

    # constraint matrix with boundary conditions
    # C_matrix = scipy.sparse.vstack((full_matrix_tet, di_matrix))

    with h5py.File(workspace_path + "matlab_input_C1_constraint_matrix.hdf5", "w") as f:
        f.create_dataset("C/values", data=np.array(full_matrix_tet.data))
        f.create_dataset("C/cols", data=np.array(full_matrix_tet.col))
        f.create_dataset("C/rows", data=np.array(full_matrix_tet.row))
        f.create_dataset(
            "C/shape",
            data=np.array([full_matrix_tet.shape[0], full_matrix_tet.shape[1]]),
        )

    with h5py.File(workspace_path + "matlab_input_dirichlet_matrix.hdf5", "w") as f:
        f.create_dataset("Di/values", data=np.array(di_matrix.data))
        f.create_dataset("Di/cols", data=np.array(di_matrix.col))
        f.create_dataset("Di/rows", data=np.array(di_matrix.row))
        f.create_dataset(
            "Di/shape", data=np.array([di_matrix.shape[0], di_matrix.shape[1]])
        )

    # expand cone constraint on tetmesh
    cone_matrix = scipy.io.mmread("CT_cone_constraint_matrix.txt")
    cone_matrix = cone_matrix.tocoo(True)
    cone_matrix_row = cone_matrix.row
    cone_matrix_col = cone_matrix.col
    cone_matrix_data = cone_matrix.data
    local2global_cone = np.array(
        local2global.tolist()
        + (local2global + v.shape[0]).tolist()
        + (local2global + v.shape[0] * 2).tolist()
    )
    # print(local2global_cone.shape)
    # print(local2global_cone)

    cone_matrix_tet_row = cone_matrix_row
    cone_matrix_tet_col = np.array(
        [local2global_cone[cone_matrix_col[i]] for i in range(cone_matrix_col.shape[0])]
    )
    cone_matrix_tet_data = cone_matrix_data
    cone_matrix_tet = scipy.sparse.coo_array(
        (cone_matrix_tet_data, (cone_matrix_tet_row, cone_matrix_tet_col)),
        shape=(cone_matrix.shape[0], v.shape[0] * 3),
    )

    with h5py.File(workspace_path + "matlab_input_cone_matrix.hdf5", "w") as f:
        f.create_dataset("C_cone/values", data=np.array(cone_matrix_tet.data))
        f.create_dataset("C_cone/cols", data=np.array(cone_matrix_tet.col))
        f.create_dataset("C_cone/rows", data=np.array(cone_matrix_tet.row))
        f.create_dataset(
            "C_cone/shape",
            data=np.array([cone_matrix_tet.shape[0], cone_matrix_tet.shape[1]]),
        )

    # exit()
    ####################################################
    #                    Call Matlab                   #
    ####################################################
    # print("[{}] ".format(datetime.datetime.now()), "start Matlab ...")
    # write_matlab_script(workspace_path + "matlab_script.m")
    # # /Applications/MATLAB_R2024b.app/bin/matlab -nojvm -nodesktop -nosplash -r c1_test_triple
    # matlab_command = path_to_matlab_exe + " -nojvm -nodesktop -nosplash -r matlab_script"
    # subprocess.run(matlab_command,  shell=True, check=True)

    # matlab_C_trip = read_matlab_sparse("matlab_C_trip.txt")
    # matlab_b = read_matlab_sparse("matlab_b.txt")
    # matlab_b = matlab_b.todense()
    # matlab_b_m = read_matlab_sparse("matlab_b_m.txt")
    # matlab_b_m = matlab_b_m.todense()
    # matlab_M = read_matlab_sparse("matlab_M.txt")

    # with h5py.File(workspace_path  + "matlab_constraint_matrix.hdf5", 'w') as f:
    #     f.create_dataset("A_triplets/values", data=matlab_C_trip.data)
    #     f.create_dataset("A_triplets/cols", data=matlab_C_trip.col.astype(np.int32))
    #     f.create_dataset("A_triplets/rows", data=matlab_C_trip.row.astype(np.int32))
    #     f.create_dataset("A_triplets/shape", data=matlab_C_trip.shape)

    #     f.create_dataset("b", data=matlab_b)

    #     f.create_dataset("b_proj", data=matlab_b_m)

    #     f.create_dataset("A_proj_triplets/values", data=matlab_M.data)
    #     f.create_dataset("A_proj_triplets/cols", data=matlab_M.col.astype(np.int32))
    #     f.create_dataset("A_proj_triplets/rows", data=matlab_M.row.astype(np.int32))
    #     f.create_dataset("A_proj_triplets/shape", data=matlab_M.shape)

    # # with h5py.File(workspace_path  + "matlab_b_matrix.hdf5", 'w') as f:
    # #     f.create_dataset("b/values", data=matlab_b.data)
    # #     f.create_dataset("b/cols", data=matlab_b.col)
    # #     f.create_dataset("b/rows", data=matlab_b.row)

    # # with h5py.File(workspace_path  + "matlab_b_m_matrix.hdf5", 'w') as f:
    # #     f.create_dataset("matlab_b_m/values", data=matlab_b_m.data)
    # #     f.create_dataset("matlab_b_m/cols", data=matlab_b_m.col)
    # #     f.create_dataset("matlab_b_m/rows", data=matlab_b_m.row)

    # # with h5py.File(workspace_path  + "matlab_M_matrix.hdf5", 'w') as f:
    # #     f.create_dataset("matlab_M/values", data=matlab_M.data)
    # #     f.create_dataset("matlab_M/cols", data=matlab_M.col)
    # #     f.create_dataset("matlab_M/rows", data=matlab_M.row)

    # print("[{}] ".format(datetime.datetime.now()), "Matlab finished.")
    ####################################################
    #                    TODO: integrate               #
    ####################################################
    # # cone constraint
    # print("[{}] ".format(datetime.datetime.now()), "constructing cone hard constraint matrix ...")
    # cone_matrix = scipy.io.mmread('CT_cone_constraint_matrix.txt')

    # v_copy = v[local2global, :]
    # v_vec = np.concatenate((v_copy[:,0], v_copy[:,1], v_copy[:,2]))
    # A_cone = cone_matrix.tocoo(True)
    # b_cone = -(A_cone @ v_vec)

    # with h5py.File(workspace_path  + "CT_cone_constraint_matrix.hdf5", 'w') as f:
    #     f.create_dataset("local2global", data=local2global.astype(np.int32)) # TODO: pending change
    #     f.create_dataset("A_triplets/values", data=A_cone.data)
    #     f.create_dataset("A_triplets/cols", data=A_cone.col)
    #     f.create_dataset("A_triplets/rows", data=A_cone.row)
    #     f.create_dataset("b", data=b_cone)

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

    # ####################################################
    # #            create json  for Polyfem              #
    # ####################################################
    print("[{}] ".format(datetime.datetime.now()), "create json for polyfem")

    # c_json = {'space': {'discr_order': 3}, 'geometry': [{'mesh': output_name + '_initial_tetmesh.msh', 'volume_selection': 1, 'surface_selection': 1}], 'constraints': {'hard': ['CT_full_constraint_matrix.hdf5'], 'soft': [{'weight': 10000.0, 'data': 'soft_1.hdf5'}, {'weight': 10000.0, 'data': 'soft_2.hdf5'}]}, 'materials': [{'id': 1, 'type': 'NeoHookean', 'E': 20000000.0, 'nu': 0.3}], 'solver': {'nonlinear': {'x_delta': 1e-10, 'solver': 'Newton', 'grad_norm': 1e-08, 'advanced': {'f_delta': 1e-10}}, 'augmented_lagrangian': {'initial_weight': 100000000.0}}, 'boundary_conditions': {'dirichlet_boundary': {'id': 1, 'value': [0, 0, 0]}}, 'output': {'paraview': {'file_name': output_name + '_final.vtu', 'surface': True, 'wireframe': True, 'points': True, 'options': {'material': True, 'force_high_order': True}, 'vismesh_rel_area': 1e-05}}}
    c_json = {}
    if not enable_offset:
        c_json = {
            "contact": {
                "dhat": 0.03,
                "enabled": False,
                "collision_mesh": {
                    "mesh": "CT_bilaplacian_nodes.obj",
                    "linear_map": "local2global_matrix.hdf5",
                },
            },
            "space": {"discr_order": 3},
            "geometry": [
                {
                    "mesh": output_name + "_initial_tetmesh.msh",
                    "volume_selection": 1,
                    "surface_selection": 1,
                },
                # {"mesh": offset_file, "is_obstacle": True},
            ],
            "constraints": {
                "hard": ["CT_constraint_with_cone_tet_ids.hdf5"],
                # "hard": ["soft_2.hdf5"],
                "soft": [
                    # {"weight": 10000000.0, "data": "CT_constraint_with_cone_tet_ids.hdf5"},
                    # {"weight": weight_soft_1, "data": "soft_1.hdf5"},
                    {"weight": weight_soft_1, "data": "soft_2.hdf5"},
                    # {"weight": weight_soft_1, "data": "soft_3.hdf5"}
                ],
            },
            "materials": [
                {"id": 1, "type": elasticity_mode, "E": 200000000000.0, "nu": 0.3}
            ],
            "boundary_conditions": {
                "dirichlet_boundary": [{"id": 1, "value": [0.0, 0.0, 0.0]}]
            },
            "solver": {
                "contact": {"barrier_stiffness": 1e8},
                "nonlinear": {
                    "first_grad_norm_tol": 0,
                    "grad_norm": 1e-06,
                    "solver": "Newton",
                    "Newton": {"residual_tolerance": 1e6},
                },
            },
            "output": {
                "paraview": {
                    "file_name": output_name + "_final.vtu",
                    "options": {"material": True, "force_high_order": True},
                    "vismesh_rel_area": 1e-05,
                }
            },
        }
    else:
        c_json = {
            "contact": {
                "dhat": 0.03,
                "enabled": True,
                "collision_mesh": {
                    "mesh": "CT_bilaplacian_nodes.obj",
                    "linear_map": "local2global_matrix.hdf5",
                },
            },
            "space": {"discr_order": 3},
            "geometry": [
                {
                    "mesh": output_name + "_initial_tetmesh.msh",
                    "volume_selection": 1,
                    "surface_selection": 1,
                },
                {"mesh": offset_file, "is_obstacle": True},
            ],
            "constraints": {
                "hard": ["CT_constraint_with_cone_tet_ids.hdf5"],
                # "hard": ["soft_2.hdf5"],
                "soft": [
                    # {"weight": 10000000.0, "data": "CT_constraint_with_cone_tet_ids.hdf5"},
                    # {"weight": weight_soft_1, "data": "soft_1.hdf5"},
                    {"weight": weight_soft_1, "data": "soft_2.hdf5"},
                    # {"weight": weight_soft_1, "data": "soft_3.hdf5"}
                ],
            },
            "materials": [
                {"id": 1, "type": elasticity_mode, "E": 200000000000.0, "nu": 0.3}
            ],
            "boundary_conditions": {
                "dirichlet_boundary": [{"id": 1, "value": [0.0, 0.0, 0.0]}]
            },
            "solver": {
                "contact": {"barrier_stiffness": 1e8},
                "nonlinear": {
                    "first_grad_norm_tol": 0,
                    "grad_norm": 1e-06,
                    "solver": "Newton",
                    "Newton": {"residual_tolerance": 1e6},
                },
            },
            "output": {
                "paraview": {
                    "file_name": output_name + "_final.vtu",
                    "options": {"material": True, "force_high_order": True},
                    "vismesh_rel_area": 1e-05,
                }
            },
        }

    
    with open("constraints.json", "w") as f:
        json.dump(c_json, f)

    # ####################################################
    # #                    Call Polyfem                  #
    # ####################################################
    print("[{}] ".format(datetime.datetime.now()), "calling polyfem")

    # print("Calling Polyfem")
    polyfem_command = path_to_polyfem_exe + " -j " + workspace_path + "constraints.json --log_level 0"
    print(polyfem_command)
    subprocess.run(polyfem_command, shell=True, check=True)

    # ####################################################
    # #             extract inside tets                  #
    # ####################################################
    polyfem_mesh = mio.read(output_name + "_final.vtu")

    new_winding_number_list = []
    for i in range(len(polyfem_mesh.cells[0])):
        assert i in new_winding_numbers
        new_winding_number_list.append(new_winding_numbers[i])
    new_winding_number_list = np.array(new_winding_number_list)

    new_winding_number_total = None
    if enable_offset:
        new_winding_number_total = np.zeros(
            (len(polyfem_mesh.cells[0]) + len(polyfem_mesh.cells[1]), 1)
        )
    else:
        new_winding_number_total = np.zeros(
            (len(polyfem_mesh.cells[0]), 1)
        )
    new_winding_number_total[: len(polyfem_mesh.cells[0])] = new_winding_number_list

    polyfem_mesh.cell_data["winding"] = new_winding_number_total[:, None]
    polyfem_mesh.write(output_name + "_final_winding.vtu")
    print("C1 Meshing DONE")
