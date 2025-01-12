import json
import sys
import os
import subprocess


run_script_path = os.path.dirname(__file__)
executables_json = os.path.join(run_script_path, "executables.json")


def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!!!! Command '{command}' failed with error: {e} !!!!!")


# read executables json
with open(executables_json, "r") as f:
    executables = json.load(f)

seamless_parametrization_binary = executables["seamless_parametrization_binary"]
polyfem_binary = executables["polyfem_binary"]
smooth_contours_binary = executables["smooth_contours_binary"]
c1_meshing_script = executables["c1_meshing_script"]

if __name__ == "__main__":
    args = sys.argv

    if len(args) < 2:
        print("Not enough arguments")
        exit()

    # print("Executables:")
    # for x in executables:
    #     print("*", x, ":", executables[x])

    input_file = args[1]
    output_name = args[2]

    if not os.path.isfile(input_file):
        print("Input file", input_file, " does not exist")
        exit()

    # compute output_name from input_file
    # create folder within working directory
    # cd to that folder and run from there

    command = " ".join(
        [
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
