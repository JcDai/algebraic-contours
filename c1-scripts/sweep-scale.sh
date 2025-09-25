#!/bin/bash

mesh=data/parametrized_splitted_meshes/greek_sculpture/surface_uv_after_cone_split.obj
output_dir=./output/fig-fitting/greek

mesh=data/parametrized_splitted_meshes/vase-lion100K/surface_uv_after_cone_split.obj
output_dir=./output/fig-fitting/vase-lion100K

mesh=data/parametrized_splitted_meshes/pegaso/surface_uv_after_cone_split.obj
output_dir=./output/fig-fitting/pegaso

mesh=data/dataset/eight/surface_uv_after_cone_split.obj
output_dir=./output/sweep/eight

for s in 1 2 3
do
	test_dir=${output_dir}/scale_${s}_output/
	mkdir -p ${test_dir}
	./build/bin/optimize_cubic_surface \
		-i ${mesh} \
		-w 1e5 \
		-n 10 \
		--scale $s \
		--invert_area \
		--log_level info \
		-o ${test_dir} \
		> ${test_dir}/optimization_log.txt &
done
