# C1 Meshing

#### Installation

The installation process was only tested on Mac.

Install and activate the conda environment

```
conda env create -f c1meshing.yml
conda activate c1meshing
```

Run the install script that downloads and installs the dependencies, and creates a JSON file with all the executables listed. **Requires CMake 3.30!**

```
python3 install_dependencies.py
```

The last line of the output should be "All executables were located. Installation successful"

#### Run

Use the script `run_c1_meshing.py`. If you want to move the script, keep the `executables.json` in the same folder!
