import c1utils

def call_CT(ct_exe, surface_file, workspace_path):
    print("[{}] ".format(datetime.datetime.now()), "Calling Clough Tocher code")
    # ct_command = path_to_ct_exe + " --input " + workspace_path + "parameterized_mesh_splitted.obj -o CT"
    ct_command = (
        ct_exe
        + " --input "
        + workspace_path
        + surface_file 
        + "-o CT"
    )

    subprocess.run(ct_command, shell=True, check=True)
    # subprocess.run(ct_command.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
