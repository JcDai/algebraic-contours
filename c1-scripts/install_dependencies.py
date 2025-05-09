import json
import os
import subprocess

install_script_path = os.path.dirname(__file__)
executables_json_filename = "executables.json"

dependencies_folder = os.path.join(install_script_path, "dependencies")
seamless_bin = os.path.join(
    dependencies_folder,
    "seamless-parametrization-penner",
    "build",
    "bin",
    "parametrize_seamless",
)
seamless_cone_gen_bin = os.path.join(
    dependencies_folder,
    "seamless-parametrization-penner",
    "build",
    "bin",
    "generate_frame_field",
)
polyfem_bin = os.path.join(dependencies_folder, "polyfem", "build", "PolyFEM_bin")
c1_meshing_script = os.path.join(
    install_script_path,
    "..",
    "src",
    "clough_tocher_patches",
    "c1_meshing_pipeline.py",
)
smooth_countours_bin = os.path.join(
    dependencies_folder, "algebraic-contours", "build", "bin", "generate_cubic_surface"
)
wmtk_bin = os.path.join(
    dependencies_folder,
    "wildmeshing-toolkit",
    "build",
    "applications",
    "tetwild_msh_converter_app",
)
wmtk_c1_bin = os.path.join(
    dependencies_folder,
    "wildmeshing-toolkit",
    "build",
    "applications",
    "c1_meshing_split_app",
)
tetwild_bin = os.path.join(dependencies_folder, "TetWild", "build", "TetWild")

print(wmtk_bin)


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

    # Seamless Parametrization
    if os.path.isfile(seamless_bin) and os.path.isfile(seamless_cone_gen_bin):
        print("=== Binary for seamless parametrization found ===")
    else:
        print("=== Installing seamless parametrization ===")
        run_command(
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

        if not os.path.isfile(seamless_bin):
            print("=== Installation of seamless parametrization failed !!! ===")

    # Algebraic Smooth Occluding Contours
    if os.path.isfile(smooth_countours_bin):
        print("=== Binary for Smooth Contours found ===")
    else:
        print("=== Installing Smooth Contours ===")
        run_command(
            """
            cd dependencies
            git clone https://github.com/JcDai/algebraic-contours.git
            cd algebraic-contours
            mkdir build
            cd build
            cmake -DCMAKE_BUILD_TYPE=Release ..
            make -j
            """
        )

        if not os.path.isfile(polyfem_bin):
            print("=== Installation of Smooth Contours failed !!! ===")

    # Polyfem
    if os.path.isfile(polyfem_bin):
        print("=== Binary for PolyFEM found ===")
    else:
        print("=== Installing PolyFEM ===")
        run_command(
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

        if not os.path.isfile(polyfem_bin):
            print("=== Installation of PolyFEM failed !!! ===")

    # Wildmeshing Toolkit
    if os.path.isfile(wmtk_bin) and os.path.isfile(wmtk_c1_bin):
        print("=== Binary for Wildmeshing Toolkit found ===")
    else:
        print("=== Installing Wildmeshing Toolkit ===")
        run_command(
            """
            cd dependencies
            git clone https://github.com/wildmeshing/wildmeshing-toolkit.git
            cd wildmeshing-toolkit
            git checkout jiacheng/c1-meshing-split
            mkdir build
            cd build
            cmake -DCMAKE_BUILD_TYPE=Release ..
            make -j
            """
        )

        if not os.path.isfile(wmtk_bin):
            print("=== Installation of Wildmeshing Toolkit failed !!! ===")

    # Tetwild
    if os.path.isfile(tetwild_bin):
        print("=== Binary for TetWild found ===")
    else:
        print("=== Installing TetWild ===")
        run_command(
            """
            cd dependencies
            git clone https://github.com/daniel-zint/TetWild.git
            cd TetWild
            mkdir build
            cd build
            cmake -DCMAKE_BUILD_TYPE=Release ..
            make -j
            """
        )

        if not os.path.isfile(tetwild_bin):
            print("=== Installation of TetWild failed !!! ===")

    if os.path.isfile(c1_meshing_script):
        print("=== Script for c1 meshing found ===")
    else:
        print("=== Script for c1 meshing NOT FOUND!!! ===")

    executable_paths = {
        "seamless_parametrization_binary": seamless_bin,
        "seamless_con_gen_binary": seamless_cone_gen_bin,
        "polyfem_binary": polyfem_bin,
        "smooth_contours_binary": smooth_countours_bin,
        "c1_meshing_script": c1_meshing_script,
        "wmtk_msh_converter_binary": wmtk_bin,
        "tetwild_binary": tetwild_bin,
        "wmtk_c1_cone_split_binary": wmtk_c1_bin,
    }

    executable_paths_json = json.dumps(executable_paths, indent=4)

    print("Generate", executables_json_filename)
    with open(executables_json_filename, "w") as f:
        f.write(executable_paths_json)

    successfull_installation = True
    for x in executable_paths:
        if os.path.isfile(executable_paths[x]):
            print("*", x, "located")
        else:
            print("*", x, "NOT FOUND!!!")
            successfull_installation = False

    if successfull_installation:
        print("All executables were located. Installation successful")
    else:
        print("Some executables were not found. INSTALLATION FAILED!!!")
