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
