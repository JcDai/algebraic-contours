#!/bin/bash

mesh=data/pegaso_surface_uv_after_cone_split.obj
output_dir=./output/fig-fitting

#for i in 6 7 8 9
for i in 10 11
do
	test_dir=${output_dir}/weight_1e${i}_output/
	mkdir -p ${test_dir}
	./build/bin/optimize_cubic_surface \
		-i ${mesh} \
		-w 1e${i} \
		-n 5 \
		--log_level info \
		-o ${test_dir} \
		> ${test_dir}/optimization_log.txt
done