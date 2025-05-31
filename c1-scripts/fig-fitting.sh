#!/bin/bash

mesh=data/parametrized_splitted_meshes/greek_sculpture/surface_uv_after_cone_split.obj
output_dir=./output/fig-fitting/greek


mesh=data/parametrized_splitted_meshes/vase-lion100K/surface_uv_after_cone_split.obj
output_dir=./output/fig-fitting/vase-lion100K

mesh=data/parametrized_splitted_meshes/pegaso/surface_uv_after_cone_split.obj
output_dir=./output/fig-fitting/pegaso

#for i in 4 5 6 7 8 9 10
for i in 2 3
do
	test_dir=${output_dir}/weight_1e${i}_output/
	mkdir -p ${test_dir}
	./build/bin/optimize_cubic_surface \
		-i ${mesh} \
		-w 1e${i} \
		-n 10 \
		--log_level info \
		-o ${test_dir} \
		> ${test_dir}/optimization_log.txt &
done
