import c1utils

def generate_highorder_tetmesh(linear_mesh_file, highorder_mesh_file, workspace_path):
    print("[{}] ".format(datetime.datetime.now()), "Calling Gmsh ... ")
    gmsh.initialize()
    gmsh.open(linear_mesh_file)
    gmsh.model.mesh.setOrder(3)
    gmsh.write(workspace_path + highorder_mesh_file)
    gmsh.write(workspace_path + "tetmesh_after_face_split_high_order_tet.m")
