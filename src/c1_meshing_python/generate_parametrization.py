import c1utils

def generate_parametrization(para_exe, workspace_path):
    print("[{}] ".format(datetime.datetime.now()), "Calling parametrization code")
    para_command = (
        para_exe
        + " --mesh "
        + workspace_path
        + "embedded_surface.obj --cones embedded_surface_Th_hat --field embedded_surface_kappa_hat"
    )
    # para_command = (
    #     path_to_para_exe
    #     + " --mesh "
    #     + workspace_path
    #     + "embedded_surface.obj --cones embedded_surface_Th_hat_reordered --field embedded_surface_kappa_hat"
    # )
    # para_command = (
    #     path_to_para_exe
    #     + " --mesh "
    #     + workspace_path
    #     + "embedded_surface.obj --cones cone_angles.txt --field embedded_surface_kappa_hat"
    # )

    subprocess.run(
        para_command,
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.DEVNULL,
    )