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


def create_polyfem_json(enable_offset, output_name, initial_mesh, soft_file, hard_file, weight_soft, elasticity_mode, offset_file):
    print("[{}] ".format(datetime.datetime.now()), "create json for polyfem")

    c_json = {}
    if not enable_offset:
        c_json = {
            "space": {
                "discr_order": 3,
                "basis_type": "Bernstein"
            },
            "geometry": [
                {
                    "mesh": initial_mesh,
                    "volume_selection": 1,
                    "surface_selection": 1,
                },
                # {"mesh": offset_file, "is_obstacle": True},
            ],
            "constraints": {
                "hard": [hard_file],
                "soft": [
                    {"weight": weight_soft, "data": soft_file},
                ],
                # "hard": [soft_file]
            },
            # "boundary_conditions": {
            #     "dirichlet_boundary": [{"id": 1, "value": [0.0, 0.0, 0.0]}]
            # },
            "materials": [
                {"id": 1, "type": elasticity_mode, "E": 200000000000.0, "nu": 0.3}
                # {"id": 1,
                #  "type": elasticity_mode,
                #  "use_rest_pose": elasticity_mode == "AMIPS", "E": 200000000000.0, "nu": 0.3}
            ],
            "solver": {
                "contact": {"barrier_stiffness": 1e8},
                "nonlinear": {
                    "first_grad_norm_tol": 0,
                    "grad_norm": 1e1,
                    "solver": "Newton",
                    "Newton": {"residual_tolerance": 1e6},
                },
            },
            "output": {
                "paraview": {
                    "file_name": output_name + "_final.vtu",
                    "options": {"material": True, "force_high_order": True},
                    "vismesh_rel_area": 1e-05,
                },
                # "advanced": {"save_solve_sequence_debug": True}
            },
            # "input": {
            #     "data": {
            #         "state": "initial_solution.hdf5",
            #         "reorder": True
            #     }
            # }
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
                # "hard": ["CT_constraint_with_cone_tet_ids.hdf5"],
                # "hard": ["soft_2.hdf5"],
                "soft": [
                    # {"weight": 10000000.0, "data": "CT_constraint_with_cone_tet_ids.hdf5"},
                    # {"weight": weight_soft_1, "data": "soft_1.hdf5"},
                    {"weight": weight_soft, "data": "soft_2.hdf5"},
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


def create_polyfem_json_amips(enable_offset, output_name, initial_mesh, soft_file, hard_file, weight_soft, elasticity_mode, offset_file):
    print("[{}] ".format(datetime.datetime.now()),
          "create amips json for polyfem")

    c_json = {}
    if not enable_offset:
        c_json = {
            "space": {
                "discr_order": 3,
                "basis_type": "Bernstein"
            },
            "geometry": [
                {
                    "mesh": initial_mesh,
                    "volume_selection": 1,
                    "surface_selection": 1,
                },
                # {"mesh": offset_file, "is_obstacle": True},
            ],
            "constraints": {
                "hard": [hard_file],
                "soft": [
                    {"weight": weight_soft, "data": soft_file},
                ],
                # "hard": [soft_file]
            },
            # "boundary_conditions": {
            #     "dirichlet_boundary": [{"id": 1, "value": [0.0, 0.0, 0.0]}]
            # },
            "materials": [
                {"id": 1,
                 "type": "AMIPS",
                 "use_rest_pose": True}
            ],
            "solver": {
                "contact": {"barrier_stiffness": 1e8},
                "nonlinear": {
                    "first_grad_norm_tol": 0,
                    "grad_norm": 1e1,
                    "solver": "Newton",
                    "Newton": {"residual_tolerance": 1e6},
                },
            },
            "output": {
                "paraview": {
                    "file_name": output_name + "_final_amips.vtu",
                    "options": {"material": True, "force_high_order": True},
                    "vismesh_rel_area": 1e-05,
                },
                # "advanced": {"save_solve_sequence_debug": True}
            },
            # "input": {
            #     "data": {
            #         "state": "initial_solution.hdf5",
            #         "reorder": True
            #     }
            # }
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
                    {"weight": weight_soft, "data": "soft_2.hdf5"},
                    # {"weight": weight_soft_1, "data": "soft_3.hdf5"}
                ],
            },
            "materials": [
                {"id": 1, "type": elasticity_mode, "E": 200000000000.0, "nu": 0.3}
            ],
            # "boundary_conditions": {
            #     "dirichlet_boundary": [{"id": 1, "value": [0.0, 0.0, 0.0]}]
            # },
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

    with open("constraints_amips.json", "w") as f:
        json.dump(c_json, f)


def call_polyfem(workspace_path, path_to_polyfem_exe, json_file):
    print("[{}] ".format(datetime.datetime.now()), "calling polyfem")

    # print("Calling Polyfem")
    polyfem_command = path_to_polyfem_exe + " -j " + \
        workspace_path + json_file + " --log_level 1"
    print(polyfem_command)
    subprocess.run(polyfem_command, shell=True, check=True)


def resurrect_winding_number(output_name, new_winding_numbers, enable_offset):
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
    new_winding_number_total[: len(
        polyfem_mesh.cells[0])] = new_winding_number_list

    polyfem_mesh.cell_data["winding"] = new_winding_number_total[:, None]
    polyfem_mesh.write(output_name + "_final_winding.vtu")
    print("C1 Meshing DONE")
