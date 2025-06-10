'''
Usage:
python compute_deviation.py <init_file.msh> <solution_file.txt> <target_file.msh> <local2global_file.txt> <b2l_mat_file.txt> <output_name>
'''


import meshio as mio
import numpy as np
import sys
import scipy

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 7:
        print("args not matching!")
        print("needed: <init_file.msh> <solution_file.txt> <target_file.msh> <local2global_file.txt> <b2l_mat_file.txt> <output_name>")

    init_file = args[1]
    solution_file = args[2]
    target_file = args[3]
    local2global_file = args[4]
    b2l_mat_file = args[5]
    output_name = args[6]

    # read initial mesh and solution
    init_mesh = mio.read(init_file)
    v_init = init_mesh.points
    t_init = init_mesh.cells_dict['tetra20']

    solution = np.loadtxt(solution_file)
    local2global = np.loadtxt(local2global_file).astype(np.int32)

    # get result beizer control points
    v_res = v_init + solution
    v_res_sf = v_res[local2global]

    # convert to lagrange
    b2l_mat = scipy.io.mmread(b2l_mat_file)
    v_res_sf_lag = b2l_mat @ v_res_sf

    # read target mesh
    target_mesh = mio.read(target_file)
    v_target = target_mesh.points

    # compute deviation
    deviation = v_res_sf_lag - v_target
    distance = np.array([np.linalg.norm(d) for d in deviation])

    max_distance = np.max(distance)
    avg_distance = np.average(distance)

    # write to file
    with open(output_name + "_max_avg_deviation.txt", "w") as file:
        file.write("{}\n".format(max_distance))
        file.write("{}\n".format(avg_distance))
