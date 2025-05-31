# Script to quadrangulate a mesh

import os, sys
base_dir = os.path.dirname(__file__)
import numpy as np
import igl
import subprocess
import argparse, shutil
import multiprocessing 

input_dir = './data/parametrized_splitted_meshes'
output_dir = './output/sweep-6'
os.makedirs(os.path.join(output_dir, 'renders'), exist_ok=True)

def process_file(m):
    # Get mesh and test name
    test_dir = os.path.join(output_dir, m + '_output')
    os.makedirs(test_dir, exist_ok=True)

    exec = ['./build/bin/optimize_cubic_surface',]
    exec += ['-i', os.path.join(input_dir, m, 'surface_uv_after_cone_split.obj')]
    exec += ['--render_path', os.path.join(output_dir, 'renders', m+'.png')]
    exec += ['-w', '1e8']
    exec += ['-n', '10']
    exec += ['--log_level', 'info']
    exec += ['--invert_area']
    exec += ['-o', test_dir]
    
    log_path = os.path.join(test_dir, 'optimization_log.txt')
    with open(log_path, 'w') as log_file:
        subprocess.run(exec, stdout=log_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--final_param",      help="number of optimization iterations",
                                                    type=int, default=0)
    args = vars(parser.parse_args())

    files = os.listdir(input_dir)
    models = [f for f in files]

    pool_args = [(m,) for m in models]
    with multiprocessing.Pool(processes=8) as pool:
        pool.starmap(process_file, pool_args, chunksize=1)

if __name__ == "__main__":
    main()
