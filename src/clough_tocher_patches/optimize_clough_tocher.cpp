#include "optimize_clough_tocher.hpp"
#include "Clough_Tocher_Laplacian.c"

/*
void CloughTocherOptimizer::initialize_ind_to_full_matrices()
{
	// TODO: Would be better to avoid the uneccesary construction of a surface
	Eigen::SparseMatrix<double> fit_matrix;
	Eigen::SparseMatrix<double> energy_hessian;
	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
			energy_hessian_inverse;
	OptimizationParameters optimization_params;
	CloughTocherSurface ct_surface(V, affine_manifold, optimization_params,
																	fit_matrix, energy_hessian,
																	energy_hessian_inverse);
	//ct_surface.m_affine_manifold.generate_lagrange_nodes();
	//ct_surface.m_affine_manifold.compute_edge_global_uv_mappings();

	// TODO: Add option to pass in
	Eigen::MatrixXd v_normals;
	igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA,
													v_normals);

	// build cone constraint system
	int64_t node_cnt = ct_surface.m_affine_manifold.m_lagrange_nodes.size();
	Eigen::SparseMatrix<double, Eigen::RowMajor> f2f_expanded(node_cnt * 3,
																														node_cnt * 3);
	f2f_expanded.reserve(Eigen::VectorXi::Constant(node_cnt * 3, 40));
	std::vector<int> independent_node_map(node_cnt * 3, -1);
	std::vector<bool> node_assigned(node_cnt, false);

	std::cout << "compute cone constraints ..." << std::endl;
	ct_surface.bezier_cone_constraints_expanded(
			f2f_expanded, independent_node_map, node_assigned, v_normals);

	std::cout << "compute endpoint constraints ..." << std::endl;
	ct_surface.bezier_endpoint_ind2dep_expanded(f2f_expanded,
																							independent_node_map, false);

	std::cout << "compute interior 1 constraints ..." << std::endl;
	ct_surface.bezier_internal_ind2dep_1_expanded(f2f_expanded,
																								independent_node_map);

	std::cout << "compute midpoint constraints ..." << std::endl;
	ct_surface.bezier_midpoint_ind2dep_expanded(f2f_expanded,
																							independent_node_map);

	std::cout << "compute interior 2 constraints ..." << std::endl;
	ct_surface.bezier_internal_ind2dep_2_expanded(f2f_expanded,
																								independent_node_map);

	std::cout << "done constraint computation" << std::endl;

	ct_surface.bezier_cone_constraints_expanded(
			f2f_expanded, independent_node_map, node_assigned, v_normals);

	// count independent variables
	int64_t dep_cnt = 0;
	int64_t ind_cnt = 0;
	for (int64_t i = 0; i < node_cnt * 3; ++i)
	{
		if (independent_node_map[i] == 0) {
			dep_cnt++;
		} else if (independent_node_map[i] == 1) {
			ind_cnt++;
		}
	}

	std::cout << "node cnt: " << node_cnt * 3 << std::endl;
	std::cout << "dep cnt: " << dep_cnt << std::endl;
	std::cout << "ind cnt: " << ind_cnt << std::endl;

	ind2full.resize(node_cnt * 3, ind_cnt);
	ind2full.reserve(Eigen::VectorXi::Constant(ind_cnt, 40));
	std::vector<int64_t> col2nid_map;
	std::vector<int64_t> ind2col_map(f2f_expanded.cols(), -1);
	col2nid_map.reserve(f2f_expanded.cols()); // preallocate space
	int64_t col_cnt = 0;
	for (int64_t i = 0; i < f2f_expanded.cols(); ++i)
	{
		if (independent_node_map[i] == 1)
		{
			const Eigen::SparseVector<double> &c = f2f_expanded.col(i);
			ind2full.col(col_cnt) = c;
			col2nid_map.push_back(i);
			ind2col_map[i] = col_cnt; // map full to independent
			col_cnt++;
		}
	}

	f2f_expanded.prune(1e-12);
	f2f_expanded.makeCompressed();
	std::vector<Triplet> ind2full_trips;
	int count = 0;
	std::vector<bool> diag_seen(f2f_expanded.rows(), false);
	std::vector<std::map<int, int>> seen_indices(f2f_expanded.rows());
	for (int k = 0; k < f2f_expanded.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(f2f_expanded, k); it; ++it)
		{
			// check if dependent node
			int j = it.col();
			if (independent_node_map[j] == 1)
			{
				if (ind2col_map[j] < 0) spdlog::error("independent column index missing for {}", j);

				// add triplet
				int i = it.row();
				double v = it.value();

				if (i == j)
				{
					if (diag_seen[i]) continue;
					diag_seen[i] = true;
				}
				if (seen_indices[i][j] > 0)
				{
					spdlog::info("Seen {} {} {} already", i, j, v);
				}
				else {
					seen_indices[i][j] = 1;
				}

				if (ind2col_map[j] == 153) spdlog::info("{} {} {} {}", count, i, j, v);
				ind2full_trips.push_back(Triplet(i, ind2col_map[j], v));
			}
			count++;
		}
	}
	Eigen::SparseMatrix<double> _ind2full;
	_ind2full.resize(node_cnt * 3, ind_cnt);
	_ind2full.setFromTriplets(ind2full_trips.begin(), ind2full_trips.end());
	Eigen::SparseMatrix<double>  diff = _ind2full - ind2full;
	spdlog::info("Matrix error is {}", diff.norm());
	for (int i = 0; i < diff.cols(); ++i)
	{
		if (diff.col(i).norm() > 1e-10)
		{
			spdlog::info("Matrix error col {} is {}", i, diff.col(i).norm());
			break;
		}
	}
	spdlog::info("Matrix error col 0 is {}", diff.col(0).norm());
	spdlog::info("Matrix error row 1 is {}", diff.row(1).norm());
	spdlog::info("Matrix entries are {} and {}", _ind2full.coeffRef(153, 153), ind2full.coeffRef(153, 153)); 

	// build projection from full to independent nodes
	std::vector<Triplet> full2ind_trips;
	for (int i = 0; i < ind_cnt; ++i)
	{
		int j = col2nid_map[i];
		full2ind_trips.push_back(Triplet(i, j, 1.));
	}
	full2ind.resize(ind_cnt, node_cnt * 3);
	full2ind.setFromTriplets(full2ind_trips.begin(), full2ind_trips.end());
}
*/

void CloughTocherOptimizer::initialize_ind_to_full_matrices()
{
	// TODO: Would be better to avoid the uneccesary construction of a surface
	Eigen::SparseMatrix<double> fit_matrix;
	Eigen::SparseMatrix<double> energy_hessian;
	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
			energy_hessian_inverse;
	OptimizationParameters optimization_params;
	CloughTocherSurface ct_surface(V, affine_manifold, optimization_params,
																	fit_matrix, energy_hessian,
																	energy_hessian_inverse);
	//ct_surface.m_affine_manifold.generate_lagrange_nodes();
	//ct_surface.m_affine_manifold.compute_edge_global_uv_mappings();

	// TODO: Add option to pass in
	Eigen::MatrixXd v_normals;
	igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA,
													v_normals);

	// build cone constraint system
	int64_t node_cnt = ct_surface.m_affine_manifold.m_lagrange_nodes.size();
	Eigen::SparseMatrix<double, Eigen::RowMajor> f2f_expanded(node_cnt * 3,
																														node_cnt * 3);
	f2f_expanded.reserve(Eigen::VectorXi::Constant(node_cnt * 3, 40));
	std::vector<int> independent_node_map(node_cnt * 3, -1);
	std::vector<bool> node_assigned(node_cnt, false);

	std::cout << "compute cone constraints ..." << std::endl;
	ct_surface.bezier_cone_constraints_expanded(
			f2f_expanded, independent_node_map, node_assigned, v_normals);

	std::cout << "compute endpoint constraints ..." << std::endl;
	ct_surface.bezier_endpoint_ind2dep_expanded(f2f_expanded,
																							independent_node_map, false);

	std::cout << "compute interior 1 constraints ..." << std::endl;
	ct_surface.bezier_internal_ind2dep_1_expanded(f2f_expanded,
																								independent_node_map);

	std::cout << "compute midpoint constraints ..." << std::endl;
	ct_surface.bezier_midpoint_ind2dep_expanded(f2f_expanded,
																							independent_node_map);

	std::cout << "compute interior 2 constraints ..." << std::endl;
	ct_surface.bezier_internal_ind2dep_2_expanded(f2f_expanded,
																								independent_node_map);

	std::cout << "done constraint computation" << std::endl;

	ct_surface.bezier_cone_constraints_expanded(
			f2f_expanded, independent_node_map, node_assigned, v_normals);

	// count independent variables
	int64_t dep_cnt = 0;
	int64_t ind_cnt = 0;
	for (int64_t i = 0; i < node_cnt * 3; ++i)
	{
		if (independent_node_map[i] == 0) {
			dep_cnt++;
		} else if (independent_node_map[i] == 1) {
			ind_cnt++;
		}
	}

	std::cout << "node cnt: " << node_cnt * 3 << std::endl;
	std::cout << "dep cnt: " << dep_cnt << std::endl;
	std::cout << "ind cnt: " << ind_cnt << std::endl;

	ind2full.resize(node_cnt * 3, ind_cnt);
	ind2full.reserve(Eigen::VectorXi::Constant(ind_cnt, 40));
	std::vector<int64_t> col2nid_map;
	std::vector<int64_t> ind2col_map(f2f_expanded.cols(), -1);
	col2nid_map.reserve(f2f_expanded.cols()); // preallocate space
	int64_t col_cnt = 0;
	for (int64_t i = 0; i < f2f_expanded.cols(); ++i)
	{
		if (independent_node_map[i] == 1)
		{
			col2nid_map.push_back(i);
			ind2col_map[i] = col_cnt; // map full to independent
			col_cnt++;
		}
	}

	std::vector<Triplet> ind2full_trips;
	std::vector<bool> diag_seen(f2f_expanded.rows(), false);
	for (int k = 0; k < f2f_expanded.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(f2f_expanded, k); it; ++it)
		{
			// check if dependent node
			int j = it.col();
			if (independent_node_map[j] == 1)
			{
				if (ind2col_map[j] < 0) spdlog::error("independent column index missing for {}", j);

				// add triplet
				int i = it.row();
				double v = it.value();

				// fix diagonal bug
				if (i == j)
				{
					if (diag_seen[i]) continue;
					diag_seen[i] = true;
				}

				ind2full_trips.push_back(Triplet(i, ind2col_map[j], v));
			}
		}
	}
	ind2full.setFromTriplets(ind2full_trips.begin(), ind2full_trips.end());

	// build projection from full to independent nodes
	std::vector<Triplet> full2ind_trips;
	for (int i = 0; i < ind_cnt; ++i)
	{
		int j = col2nid_map[i];
		full2ind_trips.push_back(Triplet(i, j, 1.));
	}
	full2ind.resize(ind_cnt, node_cnt * 3);
	full2ind.setFromTriplets(full2ind_trips.begin(), full2ind_trips.end());
}

void CloughTocherOptimizer::assemble_local_siffness_matrix(
		const std::array<PlanarPoint, 3> &face_uv_positions,
		const std::array<std::array<int64_t, 10>, 3> &patch_indices,
		std::vector<Triplet> &stiffness_matrix_trips) const
{
	double u[3];
	double v[3];
	for (int n = 0; n < 3; ++n)
	{
		u[n] = face_uv_positions[n][0];
		v[n] = face_uv_positions[n][1];
	}

	// get subtriangle uv data
	double T[3][2][2];
	compute_UV_2_bary_subtri(u, v, T);

	// get raw matrix data
	double BM[3][3][10][10];
	compute_BM(BM);

	// generate patch matrices
	double AT[3][10][10];
	compute_AT_from_uv(u, v, 0, BM, AT[0]);
	compute_AT_from_uv(u, v, 1, BM, AT[1]);
	compute_AT_from_uv(u, v, 2, BM, AT[2]);

	// compute uv triangle determinants
	double detT[3];
	for (int n = 0; n < 3; n++)
	{
		detT[n] = T[n][0][0] * T[n][1][1] - T[n][0][1] * T[n][1][0];
	}

	// need to remap from indexing assumed by
	// (a) Lagrange to Bezier conversion to (b) local stiffness matrix
	// enumerate:        0   1   2   3   4   5   6   7   8   9
	// original order: 003 300 030 102 201 210 120 021 012 111
	// new order:      003 012 021 030 102 111 120 201 210 300
	// permutation:      0   9   3   4   7   8   6   2   1   5
	std::array<int64_t, 10> perm = {0, 9, 3, 4, 7, 8, 6, 2, 1, 5};
	//std::array<int64_t, 10> perm = {0, 3, 7, 2, 3, 9, 6, 4, 5, 1};

	// assemble local matrix in global matrix
	// WARNING: renormalize by determinant first
	for (int n = 0; n < 3; n++)
	{
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				//int64_t I = patch_indices[n][perm[i]];
				//int64_t J = patch_indices[n][perm[j]];
				//double V = AT[n][i][j] / detT[n];
				int64_t I = patch_indices[n][i];
				int64_t J = patch_indices[n][j];
				double V = AT[n][perm[i]][perm[j]] / detT[n];
				stiffness_matrix_trips.push_back(Triplet(I, J, V));
			}
		}
	}
}
