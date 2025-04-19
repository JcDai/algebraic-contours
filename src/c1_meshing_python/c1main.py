from c1utils import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-j", dest="spec", required=True, type=lambda x: is_valid_json(parser, x)
    )
    parser.add_argument(
        "-b", dest="bins", required=True, type=lambda x: is_valid_json(parser, x)
    )

    args = parser.parse_args()

    input_file = args.spec[
        "input"
    ]  # vtu tetmesh file with 'winding_number' as cell data
    output_name = args.spec["output"]  # output name
    offset_file = args.spec["offset"]  # offset file
    weight_soft_1 = args.spec["weight_soft_1"]
    bilap_k_ring_neighbor = args.spec["bilap_k_ring_neighbor"] # int, bilaplacian on k ring 5 to 20
    bilap_sample_factor = args.spec["bilap_sample_factor"] # int, put 2
    elasticity_mode = args.spec["elasticity_mode"] # LinearElasticity or Neohookean
    enable_offset = args.spec["enable_offset"]

    path_to_para_exe = args.bins[
        "seamless_parametrization_binary"
    ]  # path to parametrization bin
    path_to_ct_exe = args.bins[
        "smooth_contours_binary"
    ]  # path to Clough Tocher constraints bin
    path_to_polyfem_exe = args.bins["polyfem_binary"]  # path to polyfem bin
    path_to_matlab_exe = args.bins["matlab_binary"]  # path to matlab exe
    path_to_toolkit_exe = args.bins["wmtk_c1_cone_split_binary"]  # path to toolkit app
    path_to_generate_cone_exe = args.bins["seamless_con_gen_binary"]

    

