import meshio as mio
import numpy as np
import os
import sys
import json
import subprocess

run_script_path = os.path.dirname(__file__)
executables_json = os.path.join(run_script_path, "executables.json")
to_base_json = os.path.join(run_script_path, "topological_offsets_base.json")

with open(executables_json, "r") as f:
    executables = json.load(f)

offsets_binary = executables["offsets_binary"]


def run_command(command):
    print("===== Execute:", command)
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!!!! Command '{command}' failed with error: {e} !!!!!")


def convert_to_msh(vtu_file, msh_file):
    print("Convert", vtu_file, "to", msh_file)

    m = mio.read(vtu_file)
    a = m.cell_data["winding_number"][0].copy()
    a = a - np.min(a)
    m.cell_data["in_out"] = a[:, None]

    m.cell_data = {"in_out": m.cell_data["in_out"]}
    m.write(msh_file, file_format="gmsh", binary=True)


if __name__ == "__main__":
    args = sys.argv

    if len(args) < 1:
        print("Not enough arguments")
        exit()

    input_file = args[1]

    if not os.path.isfile(input_file):
        print("Input file", input_file, " does not exist")
        exit()

    input_file_stem = os.path.splitext(input_file)[0]
    input_file_basename = os.path.basename(input_file)
    input_file_basename_stem = os.path.splitext(input_file_basename)[0]
    to_json_filename = input_file_basename_stem + "_offsets.json"

    msh_file = input_file_basename_stem + ".msh"

    convert_to_msh(input_file, msh_file)

    with open(to_base_json, "r") as f:
        to_base = json.load(f)

    to_base["input"] = msh_file
    to_base_json = json.dumps(to_base, indent=4)

    print("Write", to_json_filename)
    with open(to_json_filename, "w") as f:
        f.write(to_base_json)

    command = " ".join(
        [
            offsets_binary,
            "-j",
            to_json_filename,
        ]
    )
    run_command(command)
