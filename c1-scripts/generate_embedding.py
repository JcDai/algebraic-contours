import json
import sys
import os
import subprocess
import traceback
import shutil

run_script_path = os.path.dirname(__file__)
executables_json = os.path.join(run_script_path, "executables.json")


def run_command(command):
    print("===== Execute:", command)
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!!!! Command '{command}' failed with error: {e} !!!!!")


# read executables json
with open(executables_json, "r") as f:
    executables = json.load(f)

tetwild_binary = executables["tetwild_binary"]
wmtk_msh_converter_binary = executables["wmtk_msh_converter_binary"]

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

    tmp_folder = input_file_basename_stem

    # create folder
    try:
        os.makedirs(tmp_folder, exist_ok=False)
    except FileExistsError as e:
        traceback.print_exc()
        print(
            "Folder",
            tmp_folder,
            "already exists and would be overwritten. Delete the folder to run this script",
        )
        exit()

    # copy input file into folder
    shutil.copyfile(input_file, os.path.join(tmp_folder, input_file_basename))

    # TetWild
    command = " ".join(
        [
            "cd",
            tmp_folder,
            "; ",
            tetwild_binary,
            "--input",
            input_file,
            "--save-mid-result 2",
        ]
    )
    run_command(command)

    tetwild_output_msh = input_file_basename_stem + "__mid2.000000.msh"
    tetwild_output_obj = input_file_basename_stem + "__sf.obj"

    # wmtk VTU conversion
    command = " ".join(
        [
            "cd",
            tmp_folder,
            "; ",
            wmtk_msh_converter_binary,
            "-i",
            tetwild_output_msh,
        ]
    )
    run_command(command)

    wmtk_output = os.path.splitext(tetwild_output_msh)[0] + "_tets.vtu"
    print("wmtk output:", wmtk_output)
    # copy output files
    shutil.copyfile(
        os.path.join(tmp_folder, wmtk_output),
        input_file_basename_stem + "_tets.vtu",
    )
    shutil.copyfile(
        os.path.join(tmp_folder, tetwild_output_obj),
        tetwild_output_obj,
    )

    # remove tmp folder
    shutil.rmtree(input_file_basename_stem)
