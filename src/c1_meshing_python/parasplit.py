import c1utils

def para_split(tets, vertices, surface_adj_tet, tet_surface, para_before_file, para_after_file):
    v_before, _, _, f_before, _, _ = igl.read_obj(para_before_file)
    v_after, _, _, f_after, _, _ = igl.read_obj(para_after_file)

    adj_before = igl.adjacency_list(f_before)
    adj_after = igl.adjacency_list(f_after) 

    v_new = list(range(v_before.shape[0], v_after.shape[0]))

    v_process = copy.deepcopy(v_new)
    v_rest = copy.deepcopy(v_process)

    edges = {}
    for v in range(len(adj_before)):
        for v_adj in adj_before[v]:
            edges[str(v) + "+" + str(v_adj)] = True

    edges_to_split = []
    while v_rest:
        v_process = v_rest
        v_rest = []
        for v in v_process:
            assert len(adj_after[v]) == 4
            old_cnt = 0
            for v_adj in adj_after[v]:
                if v_adj not in v_process:
                    old_cnt += 1
            if old_cnt > 0:
                assert old_cnt == 2 or old_cnt == 4
                # find old edges
                old_endpoints = []
                for v_adj in adj_after[v]:
                    if str(v) + "+" + str(v_adj) in edges:
                        old_endpoints.append(v_adj)
                assert len(old_endpoints) == 2
                edges_to_split.append(old_endpoints)
                # add new edges to edges dict
                for v_adj in adj_after[v]:
                    edges[str(v) + "+" + str(v_adj)] = True
                    edges[str(v_adj) + "+" + str(v)] = True
            else:
                v_rest.append(v)

    print(edges_to_split)

    # TODO: add toolkit part
        
            
