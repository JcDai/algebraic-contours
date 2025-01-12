import json
import sys
import os
import subprocess
import traceback


run_script_path = os.path.dirname(__file__)
executables_json = os.path.join(run_script_path, "executables.json")


def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!!!! Command '{command}' failed with error: {e} !!!!!")


if __name__ == "__main__":
    args = sys.argv

    if len(args) < 1:
        print("Not enough arguments")
        exit()

    input_file = args[1]

    input_file_stem = os.path.splitext(input_file)[0]
    input_file_basename = os.path.basename(input_file)
    input_file_basename_stem = os.path.splitext(input_file_basename)[0]

    input_file = os.path.abspath(input_file)

    if len(args) > 2:
        output_name = args[2]
    else:
        output_name = input_file_basename_stem

    if not os.path.isfile(input_file):
        print("Input file", input_file, " does not exist")
        exit()

    ###########################
    # read executables json
    with open(executables_json, "r") as f:
        executables = json.load(f)

    seamless_parametrization_binary = executables["seamless_parametrization_binary"]
    polyfem_binary = executables["polyfem_binary"]
    smooth_contours_binary = executables["smooth_contours_binary"]
    c1_meshing_script = executables["c1_meshing_script"]

    # create folder within working directory
    os.makedirs(output_name, exist_ok=True)

    # cd to that folder and run from there
    command = " ".join(
        [
            "cd",
            output_name,
            "; ",
            "python3",
            c1_meshing_script,
            input_file,
            output_name,
            seamless_parametrization_binary,
            smooth_contours_binary,
            polyfem_binary,
        ]
    )
    run_command(command)
