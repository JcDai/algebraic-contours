import c1utils

def generate_framefield(generate_cone_exe, workspace_path):
    print("[{}] ".format(datetime.datetime.now()), "Calling generate frame field code")
    para_command = (
        generate_cone_exe
        + " --mesh "
        + workspace_path
        + "embedded_surface.obj --input ./"
    )
    # para_command = path_to_generate_cone_exe + " --mesh " + workspace_path + "tet.obj --input ./"

    subprocess.run(para_command, shell=True, check=True)

    # rearrange cones
    angles = np.loadtxt("embedded_surface_Th_hat")

    cone_vids = []
    for i, angle in enumerate(angles):
        if angle < 6.0 or angle > 6.5:  # less or more than 2 * pi
            cone_vids.append(i)
    print("cone cnt: ", len(cone_vids))

    # TODO: set picked cones

    # v_embd, _, _, f_embd, _, _ = igl.read_obj("embedded_surface.obj")

    # adjlist = igl.adjacency_list(f_embd)
    # colored = [False] * v_embd.shape[0]
    # cone_list = copy.deepcopy(cone_vids)

    # cannot_place_id = []
    # placed_id = []
    # for c in cone_list:
    #     if not colored[c]:
    #         placed_id.append(c)
    #         colored[c] = True
    #         for adjv in adjlist[c]:
    #             colored[adjv] = True
    #     else:
    #         cannot_place_id.append(c)

    # exchange_map = {}
    # second_failed = []

    # for c in cannot_place_id:
    #     # try place in one ring
    #     found = False
    #     for adjv in adjlist[c]:
    #         if not colored[adjv]:
    #             exchange_map[c] = adjv
    #             exchange_map[adjv] = c
    #             colored[adjv] = True
    #             for vv in adjlist[adjv]:
    #                 colored[vv] = True
    #             found = True
    #             break
    #     if not found:
    #         second_failed.append(c)

    # third_failed = []
    # for c in second_failed:
    #     found = False
    #     for vv in range(v_embd.shape[0]):
    #         if not colored[vv]:
    #             exchange_map[c] = vv
    #             exchange_map[vv] = c
    #             colored[vv] = True
    #             for vvv in adjlist[vv]:
    #                 colored[vvv] = True
    #             found = True
    #             break
    #     if not found:
    #         third_failed.append(c)

    # if len(third_failed) != 0:
    #     print("cannot easily place cone! try make mesh denser")
    #     exit()

    # reorder_vid = []
    # for i in range(v_embd.shape[0]):
    #     if i not in exchange_map:
    #         reorder_vid.append(i)
    #     else:
    #         reorder_vid.append(exchange_map[i])

    # assert len(reorder_vid) == v_embd.shape[0]

    # with open("embedded_surface_Th_hat_reordered", "w") as file:
    #     for i in range(v_embd.shape[0]):
    #         file.write("{}\n".format(angles[reorder_vid[i]]))

    # angles_new = np.loadtxt("embedded_surface_Th_hat_reordered")
    # cone_vids_new = []
    # for i, angle_new in enumerate(angles_new):
    #     if angle_new < 6.0 or angle_new > 6.5:  # less or more than 2 * pi
    #         cone_vids_new.append(i)

    # assert len(cone_vids_new) == len(cone_vids)

    return cone_vids