import json
import os
import subprocess
import multiprocessing

install_script_path = os.path.dirname(__file__)
executables_json_filename = "executables.json"


def run_command(command, logging=True):
    try:
        print(f"***** Execute: {command}")
        if logging:
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        print(f"***** Success")
    except subprocess.CalledProcessError as e:
        print(f"!!!!! Command '{command}' failed with error: {e} !!!!!")


if __name__ == "__main__":
    # change working directory to install_script_path
    os.chdir(install_script_path)
    print("Working directory: ", os.getcwd())

    # create dependencies folder
    os.makedirs("dependencies", exist_ok=True)

    commands = []
    # Seamless Parametrization
    commands.append(
        """
        cd dependencies
        git clone --recurse-submodules https://github.com/rjc8237/seamless-parametrization-penner.git
        cd seamless-parametrization-penner
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j
        """
    )
    # Polyfem
    commands.append(
        """
        cd dependencies
        git clone https://github.com/polyfem/polyfem.git
        cd polyfem
        git checkout -b generic_al origin/generic_al
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j
        """
    )

    for command in commands:
        run_command(command)

    # get all paths to executables
    dependencies_folder = os.path.join(install_script_path, "dependencies")
    seamless_bin = os.path.join(
        dependencies_folder,
        "seamless-parametrization-penner",
        "build",
        "bin",
        "parametrize_seamless",
    )
    print("Path seamless parametrization:", seamless_bin)
    if os.path.isfile(seamless_bin):
        print("\t=== Binary for seamless parametrization found ===")
    else:
        print("\t=== Binary for seamless parametrization NOT FOUND!!! ===")

    polyfem_bin = os.path.join(dependencies_folder, "polyfem", "build", "PolyFEM_bin")
    print("Path PolyFEM:", polyfem_bin)
    if os.path.isfile(polyfem_bin):
        print("\t=== Binary for PolyFEM found ===")
    else:
        print("\t=== Binary for PolyFEM NOT FOUND!!! ===")

    c1_meshing_script = os.path.join(
        install_script_path,
        "..",
        "src",
        "clough_tocher_patches",
        "c1_meshing_pipeline.py",
    )
    print("Path c1 meshing script:", c1_meshing_script)
    if os.path.isfile(c1_meshing_script):
        print("\t=== Script for c1 meshing found ===")
    else:
        print("\t=== Script for c1 meshing NOT FOUND!!! ===")

    executable_paths = {
        "seamless_parametrization_binary": seamless_bin,
        "polyfem_binary": polyfem_bin,
        "c1_meshing_script": c1_meshing_script,
    }

    executable_paths_json = json.dumps(executable_paths, indent=4)

    print("Generate", executables_json_filename)
    with open(executables_json_filename, "w") as f:
        f.write(executable_paths_json)
