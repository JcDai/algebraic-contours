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


def call_CT_code(workspace_path, path_to_ct_exe, meshfile, skip_cons=False, use_initial_guess=False):
    print("[{}] ".format(datetime.datetime.now()),
          "Calling Clough Tocher code")

    ct_command = ""
    if skip_cons:
        ct_command = (
            path_to_ct_exe
            + " --input "
            + workspace_path
            + meshfile + " -o CT --skip_constraint true"
        )
    else:
        ct_command = (
            path_to_ct_exe
            + " --input "
            + workspace_path
            + meshfile + " -o CT"
        )

    if use_initial_guess:
        ct_command += " --use_incenter"

    subprocess.run(ct_command, shell=True, check=True)


def call_CT_optimize_code(workspace_path, path_to_ct_optimize_exe, meshfile, ct_weight, ct_iteration, use_initial_guess=False):
    print("[{}] ".format(datetime.datetime.now()),
          "Calling Clough Tocher cubic optimization code")
    ct_command = (
        path_to_ct_optimize_exe
        + " --input "
        + workspace_path
        + meshfile + " -o CT -w " + str(ct_weight) + " -n " + str(ct_iteration)
    )

    if use_initial_guess:
        ct_command += " --use_incenter"

    subprocess.run(ct_command, shell=True, check=True)
