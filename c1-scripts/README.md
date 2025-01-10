# C1 Meshing

### Requirements / Dependencies

**Requires CMake 3.30**

- Parametrization: https://github.com/rjc8237/seamless-parametrization-penner.git
  - binary: ./build/bin/parametrize_seamless
- Polyfem: https://github.com/polyfem/polyfem.git
  - branch: generic_al
  - binary: ./build/PolyFEM_bin

Required Python packages:

- igl
- meshio
- numpy
- copy
- subprocess
- sys
- gmsh
- h5py
- scipy
- json

#### Installing dependencies

Parametrization

```
git clone --recurse-submodules https://github.com/rjc8237/seamless-parametrization-penner.git
cd seamless-parametrization-penner/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4
```

Polyfem

```
git clone https://github.com/polyfem/polyfem.git
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4
```
