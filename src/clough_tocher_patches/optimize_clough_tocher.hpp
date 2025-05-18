#pragma once

#include "common.h"
#include "affine_manifold.h"
#include "clough_tocher_surface.hpp"

// TODO: Move to cpp file
#include <igl/per_vertex_normals.h>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/CholmodSupport>
#include <Eigen/Sparse>

class CloughTocherOptimizer
{
public:
	CloughTocherOptimizer();

  std::vector<Eigen::Vector3d>& get_bezier_control_points() { return m_bezier_control_points; }
  const std::vector<Eigen::Vector3d>& get_bezier_control_points() const { return m_bezier_control_points; }

	Eigen::SparseMatrix<double>& get_stiffness_matrix() { return m_stiffness_matrix; }
	const Eigen::SparseMatrix<double>& get_stiffness_matrix() const { return m_stiffness_matrix; }

private:
	Eigen::SparseMatrix<double> m_stiffness_matrix;
	Eigen::SparseMatrix<double> m_constraint;
  std::vector<Eigen::Vector3d> m_bezier_control_points;
	AffineManifold affine_manifold;
  Eigen::MatrixXd V, uv, N;
  Eigen::MatrixXi F, FT, FN;

	Eigen::SparseMatrix<double> generate_ind_to_full_matrix()
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
		ct_surface.bezier_cone_constraints_expanded(
				f2f_expanded, independent_node_map, node_assigned, v_normals);

		// count independent variables
		int64_t ind_cnt = 0;
		for (int64_t i = 0; i < node_cnt * 3; ++i) {
			if (independent_node_map[i] == 1) {
				ind_cnt++;
			}
		}

		Eigen::SparseMatrix<double> r2f_expanded(node_cnt * 3, ind_cnt);
		r2f_expanded.reserve(Eigen::VectorXi::Constant(ind_cnt, 40));
		std::vector<int64_t> col2nid_map;
		int64_t col_cnt = 0;
		for (int64_t i = 0; i < f2f_expanded.cols(); ++i) {
			if (independent_node_map[i] == 1) {
				const Eigen::SparseVector<double> &c = f2f_expanded.col(i);
				r2f_expanded.col(col_cnt) = c;
				col_cnt++;
				col2nid_map.push_back(i);
			}
		}

		return r2f_expanded;
	}

	
	typedef Eigen::Matrix<double, 10, 10> CubicHessian;
	CubicHessian generate_local_siffness_matrix() const
	{
			// TODO: Need to generate patch geometry
			return CubicHessian();
	}

	/**
	 * @brief Get the Bezier node indices of the three micro-triangles of a given face.
	 * 
	 * @param face_index: mesh face index
	 * @return array of 3 patch Bezier nodes (10 node indices per patch)
	 */
  std::array<std::array<int64_t, 10>, 3> get_micro_triangle_nodes(int64_t face_index)
	{
		FaceManifoldChart face_chart = affine_manifold.get_face_chart(face_index);

		// macro node indexing reference is generate_lagrange_nodes in affine_manifold.cpp
		const auto& macro_nodes = face_chart.lagrange_nodes;

		// repackage macro-triangle node indices into micro-triangle patch indices
		// TODO: determine final ordering from autogen code
		std::array<std::array<int64_t, 10>, 3> micro_nodes;
		for (int i = 0; i < 3; ++i)
		{
			// get next index mod 3
			int j = (i + 1) % 3;

			// vertex nodes
			micro_nodes[i][0] = macro_nodes[i];
			micro_nodes[i][1] = macro_nodes[j];
			micro_nodes[i][2] = macro_nodes[18];

			// exterior edge nodes
			int ext_edge_offset = 3;
			int ext_edge_incr = 2;
			micro_nodes[i][3] = macro_nodes[ext_edge_offset + (ext_edge_incr * i)];
			micro_nodes[i][4] = macro_nodes[ext_edge_offset + (ext_edge_incr * i) + 1];

			// edge nodes
			int int_edge_offset = 12;
			int int_edge_incr = 2;
			micro_nodes[i][5] = macro_nodes[int_edge_offset + (int_edge_incr * i)];
			micro_nodes[i][6] = macro_nodes[int_edge_offset + (int_edge_incr * i) + 1];
			micro_nodes[i][7] = macro_nodes[int_edge_offset + (int_edge_incr * j)];
			micro_nodes[i][8] = macro_nodes[int_edge_offset + (int_edge_incr * j) + 1];

			// center node
			micro_nodes[i][9] = macro_nodes[i + 9];
		}

		return micro_nodes;
	}

	typedef Eigen::Triplet<double> Triplet;

	/**
	 * @brief Assemble local hessian into the global Hessian.
	 * 
	 * @param patch_indices: mapping from local to global indices
	 * @param local_hessian: local Hessian
	 * @param global_hessian_trips: global Hessian (represented with IJV triplets)
	 */
  void assemble_patch_coefficients(
		const std::array<int64_t, 10>& patch_indices,
		const CubicHessian& local_hessian,
		std::vector<Triplet>& global_hessian_trips)
	{
		for (int i = 0; i < 10; ++i)
		{
			for (int j = 0; j < 10; ++j)
			{
				int I = patch_indices[i];
				int J = patch_indices[j];
				double V = local_hessian(i, j);
				global_hessian_trips.push_back(Triplet(I, J, V));
			}
		}
	}

	Eigen::SparseMatrix<double> generate_stiffness_matrix()
	{
		// assemble IJV matrix entries
		std::vector<Triplet> stiffness_matrix_trips;
		int num_faces = affine_manifold.num_faces();
		for (int fijk = 0; fijk < num_faces; ++fijk)
		{
			std::array<std::array<int64_t, 10>, 3> nodes = get_micro_triangle_nodes(fijk);
			for (int i = 0; i < 3; ++i)
			{
				CubicHessian local_stiffness_matrix = generate_local_siffness_matrix(); // TODO
				assemble_patch_coefficients(nodes[i], local_stiffness_matrix, stiffness_matrix_trips);
			}
		}

		// build matrix
		int node_cnt = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
		Eigen::SparseMatrix<double> stiffness_matrix;
		stiffness_matrix.resize(node_cnt, node_cnt);
		stiffness_matrix.setFromTriplets(stiffness_matrix_trips.begin(), stiffness_matrix_trips.end());
		
		return stiffness_matrix;
	}


	void optimize_energy()
	{
		// compute hessian
		Eigen::SparseMatrix<double> C = generate_ind_to_full_matrix();
		Eigen::SparseMatrix<double> A = get_stiffness_matrix();
		Eigen::SparseMatrix<double> hessian = C.transpose() * (A * C);

		// invert hessian
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> hessian_inverse;
		hessian_inverse.compute(hessian);

		// get derivative

	} 

	
};