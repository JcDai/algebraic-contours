#pragma once

#include "common.h"
#include "affine_manifold.h"
#include "clough_tocher_surface.hpp"

// TODO: Move to cpp file
#include <igl/per_vertex_normals.h>
#include <igl/Timer.h>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/CholmodSupport>
#include <Eigen/Sparse>

class CloughTocherOptimizer
{
public:
	std::vector<Eigen::Vector3d> &get_bezier_control_points() { return m_bezier_control_points; }
	const std::vector<Eigen::Vector3d> &get_bezier_control_points() const { return m_bezier_control_points; }

	Eigen::SparseMatrix<double> &get_ind_to_full_matrix() { return ind2full; };
	Eigen::SparseMatrix<double> &get_full_to_ind_matrix() { return full2ind; };
	double fitting_weight;

private:
	typedef Eigen::Matrix<double, 10, 10> CubicHessian;
	typedef Eigen::Triplet<double> Triplet;

	Eigen::SparseMatrix<double> m_stiffness_matrix;
	Eigen::SparseMatrix<double> m_constraint;
	std::vector<Eigen::Vector3d> m_bezier_control_points;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	AffineManifold affine_manifold;

	Eigen::SparseMatrix<double> full2ind, ind2full;

	Eigen::SparseMatrix<double> triple_matrix(const Eigen::SparseMatrix<double>& mat) const
	{
		int rows = mat.rows();
		std::vector<Triplet> matrix_trips;
		for (int k = 0; k < mat.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
			{
				int i = it.row();
				int j = it.col();
				double v = it.value();
				for (int n = 0; n < 3; ++n)
				{
					int I = (i * 3) + n;
					int J = (j * 3) + n;
					matrix_trips.push_back(Triplet(I, J, v));
				}
			}
		}

		Eigen::SparseMatrix<double> tripled_mat;
		tripled_mat.resize(3 * rows, 3 * rows);
		tripled_mat.setFromTriplets(matrix_trips.begin(), matrix_trips.end());

		return tripled_mat;
	}

	void initialize_ind_to_full_matrices();

	void assemble_local_siffness_matrix(
			const std::array<PlanarPoint, 3> &face_uv_positions,
			const std::array<std::array<int64_t, 10>, 3> &patch_indices,
			std::vector<Triplet> &stiffness_matrix_trips) const;

	/**
	 * @brief Get the Bezier node indices of the three micro-triangles of a given face.
	 *
	 * @param face_index: mesh face index
	 * @return array of 3 patch Bezier nodes (10 node indices per patch)
	 */
	std::array<std::array<int64_t, 10>, 3> _get_micro_triangle_nodes(int64_t face_index)
	{
		FaceManifoldChart face_chart = affine_manifold.get_face_chart(face_index);

		// macro node indexing reference is generate_lagrange_nodes in affine_manifold.cpp
		const auto &macro_nodes = face_chart.lagrange_nodes;

		// repackage macro-triangle node indices into micro-triangle patch indices
		// TODO: determine final ordering from autogen code
		// WARNING: need to convert from Lagrange node to Bezier node
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

	// in order: 003 300 030 102 201 210 120 021 012 111
	// TODO: taken from clough_tocher_surface.cpp; make a common reference
	std::array<std::array<int64_t, 10>, 3> get_micro_triangle_nodes(int64_t face_index)
	{
		FaceManifoldChart face_chart = affine_manifold.get_face_chart(face_index);
		const auto &l_nodes = face_chart.lagrange_nodes;
		return {
				{{{l_nodes[0], l_nodes[1], l_nodes[18], l_nodes[3], l_nodes[4],
					 l_nodes[14], l_nodes[15], l_nodes[13], l_nodes[12], l_nodes[9]}},
				 {{l_nodes[1], l_nodes[2], l_nodes[18], l_nodes[5], l_nodes[6],
					 l_nodes[16], l_nodes[17], l_nodes[15], l_nodes[14], l_nodes[10]}},
				 {{l_nodes[2], l_nodes[0], l_nodes[18], l_nodes[7], l_nodes[8],
					 l_nodes[12], l_nodes[13], l_nodes[17], l_nodes[16], l_nodes[11]}}}};
	}

	/**
	 * @brief Assemble local hessian into the global Hessian.
	 *
	 * @param patch_indices: mapping from local to global indices
	 * @param local_hessian: local Hessian
	 * @param global_hessian_trips: global Hessian (represented with IJV triplets)
	 */
	void assemble_patch_coefficients(
			const std::array<int64_t, 10> &patch_indices,
			const CubicHessian &local_hessian,
			std::vector<Triplet> &global_hessian_trips)
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

	Eigen::SparseMatrix<double> get_stiffness_matrix()
	{
		// assemble IJV matrix entries
		std::vector<Triplet> stiffness_matrix_trips;
		int num_faces = affine_manifold.num_faces();
		for (int fijk = 0; fijk < num_faces; ++fijk)
		{
			FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
			std::array<std::array<int64_t, 10>, 3> nodes = get_micro_triangle_nodes(fijk);
			assemble_local_siffness_matrix(face_chart.face_uv_positions, nodes, stiffness_matrix_trips);
		}

		// build matrix
		int node_cnt = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
		Eigen::SparseMatrix<double> stiffness_matrix;
		stiffness_matrix.resize(node_cnt, node_cnt);
		stiffness_matrix.setFromTriplets(stiffness_matrix_trips.begin(), stiffness_matrix_trips.end());

		return triple_matrix(stiffness_matrix);
	}

	Eigen::SparseMatrix<double> get_position_matrix()
	{
		// get list of position vertex indices
		int num_faces = affine_manifold.num_faces();
		int num_nodes = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
		std::vector<bool> is_vertex_node(num_nodes, false);
		for (int fijk = 0; fijk < num_faces; ++fijk)
		{
			FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
			const auto &nodes = face_chart.lagrange_nodes;
			is_vertex_node[nodes[0]] = true;
			is_vertex_node[nodes[1]] = true;
			is_vertex_node[nodes[2]] = true;
		}

		// assemble IJV matrix entries
		std::vector<Triplet> position_matrix_trips;
		for (int i = 0; i < num_nodes; ++i)
		{
			if (!is_vertex_node[i])
				continue;
			position_matrix_trips.push_back(Triplet(i, i, 1.));
		}

		// build matrix
		Eigen::SparseMatrix<double> position_matrix;
		position_matrix.resize(num_nodes, num_nodes);
		position_matrix.setFromTriplets(position_matrix_trips.begin(), position_matrix_trips.end());

		return triple_matrix(position_matrix);
	}

	std::vector<int> independent_node_map;
	int count_nodes() const { return independent_node_map.size(); }

	Eigen::VectorXd build_node_vector(const std::vector<Eigen::Vector3d> &bezier_control_points)
	{
		int num_nodes = bezier_control_points.size();
		Eigen::VectorXd p(3 * num_nodes);
		for (int i = 0; i < num_nodes; ++i)
		{
			for (int n = 0; n < 3; ++n)
			{
				p[(3 * i) + n] = bezier_control_points[i][n];
			}
		}

		return p;
	}

	std::vector<Eigen::Vector3d> build_control_points(const Eigen::VectorXd &p)
	{
		int num_nodes = p.size() / 3;
		std::vector<Eigen::Vector3d> bezier_control_points(num_nodes);
		for (int i = 0; i < num_nodes; ++i)
		{
			for (int n = 0; n < 3; ++n)
			{
				bezier_control_points[i][n] = p[(3 * i) + n];
			}
		}

		return bezier_control_points;
	}

public:
	igl::Timer timer;

	CloughTocherOptimizer(
			const Eigen::MatrixXd _V,
			const Eigen::MatrixXi _F,
			const AffineManifold _affine_manifold)
			: fitting_weight(1e5), V(_V), F(_F), affine_manifold(_affine_manifold)
	{
		timer.start();	
		initialize_ind_to_full_matrices();
		spdlog::info("constraint matrix construction took {} s", timer.getElapsedTime());
	}

	double evaluate(
		const Eigen::SparseMatrix<double>& H,
		const Eigen::VectorXd& d,
		const double& E0,
		const Eigen::VectorXd& N
	) {
		double energy = 0.;
		energy += 0.5 * N.dot(H * N);
		energy += d.dot(N);
		energy += E0;
		return energy;
	}

	std::vector<Eigen::Vector3d> optimize_energy(const std::vector<Eigen::Vector3d> &bezier_control_points)
	{
		// build initial position vector
		Eigen::VectorXd p0 = build_node_vector(bezier_control_points);
		igl::Timer timer;
		timer.start();	


		// compute hessian
		double k = fitting_weight;

		const Eigen::SparseMatrix<double>& C = get_ind_to_full_matrix();
		const Eigen::SparseMatrix<double>& F = get_full_to_ind_matrix();

		Eigen::SparseMatrix<double> A = get_stiffness_matrix();
		Eigen::SparseMatrix<double> P = get_position_matrix();
		//Eigen::SparseMatrix<double> hessian = C.transpose() * C;
		Eigen::SparseMatrix<double> hessian = C.transpose() * ((A + k * P) * C);
		spdlog::info("matrix construction took {} s", timer.getElapsedTime());

		// invert hessian
		Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> hessian_inverse;
		hessian_inverse.compute(hessian);
		spdlog::info("matrix solve took {} s", timer.getElapsedTime());

		// get base energy
		double E0 = 0.5 * k * p0.dot(p0);

		// get derivative
		//Eigen::VectorXd derivative = -C.transpose() * p0;
		Eigen::VectorXd derivative = -k * C.transpose() * (P * p0);

		// print initial energy
		Eigen::VectorXd N0 = F * p0;
		spdlog::info("initial energy is {}", evaluate(hessian, derivative, E0, N0));

		// solve for optimal solution
		Eigen::VectorXd N = -hessian_inverse.solve(derivative);
		Eigen::VectorXd p = C * N;
		Eigen::VectorXd res = (hessian * N) + derivative;
		spdlog::info("optimized energy is {}", evaluate(hessian, derivative, E0, N));
		spdlog::info("residual error is {}", res.cwiseAbs().maxCoeff());
		Eigen::VectorXd pr = C * (F * p);
		spdlog::info("constraint reconstruction error is {}", (pr - p).cwiseAbs().maxCoeff());
		spdlog::info("solution assembly took {} s", timer.getElapsedTime());
		timer.stop();

		return build_control_points(p);
	}

	double evaluate_full(const std::vector<Eigen::Vector3d> &bezier_control_points)
	{
		// build initial position vector
		Eigen::VectorXd p0 = build_node_vector(bezier_control_points);

		// compute hessian
		double k = fitting_weight;
		Eigen::SparseMatrix<double> A = get_stiffness_matrix();
		Eigen::SparseMatrix<double> P = get_position_matrix();
		Eigen::SparseMatrix<double> hessian = (A + k * P);

		// invert hessian
		Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> hessian_inverse;
		hessian_inverse.compute(hessian);

		// get base energy
		double E0 = 0.5 * k * p0.dot(p0);

		// get derivative
		//Eigen::VectorXd derivative = -C.transpose() * p0;
		Eigen::VectorXd derivative = -k * (P * p0);

		// print initial energy
		double E = evaluate(hessian, derivative, E0, p0);
		spdlog::info("initial energy is {}", E);

		return E;
	}
};