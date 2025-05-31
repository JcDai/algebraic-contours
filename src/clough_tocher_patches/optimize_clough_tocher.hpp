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
	/**
	 * @brief Initialize the optimizer with the triangle mesh data.
	 *
	 * @param V: mesh vertices
	 * @param F: mesh faces
	 * @param affine_manifold: mesh topology and affine manifold structure
	 */
	CloughTocherOptimizer(
			const Eigen::MatrixXd V,
			const Eigen::MatrixXi F,
			const AffineManifold affine_manifold);

	/**
	 * @brief Optimize the quadratic Laplacian energy over the parameterization metric with fitting term
	 * while maintining the C1 constraints.
	 *
	 * @param bezier_control_points: initial Bezier points (including positions for fitting term)
	 * @return optimized control points
	 */
	std::vector<Eigen::Vector3d> optimize_laplacian_energy(const std::vector<Eigen::Vector3d> &bezier_control_points);

	/**
	 * @brief Optimize the Laplace Beltrami energy with fitting term, starting over the inital surface metric,
	 * while maintining the C1 constraints.
	 *
	 * @param bezier_control_points: initial Bezier points (including positions for fitting term)
	 * @param iterations: number of iterations of metric optimization to apply
	 * @return optimized control points
	 */
	std::vector<Eigen::Vector3d> optimize_laplace_beltrami_energy(const std::vector<Eigen::Vector3d> &bezier_control_points, int iterations=1);

	/**
	 * @brief Evaluate the quadratic Laplacian energy over the parameterization metric with fitting term.
	 *
	 * @param bezier_control_points: Bezier points
	 * @return energy for the Bezier points
	 */
	double evaluate_energy(const std::vector<Eigen::Vector3d> &bezier_control_points);

	Eigen::SparseMatrix<double> generate_laplace_beltrami_stiffness_matrix() const;

	std::vector<double> compute_face_energies(const std::vector<Eigen::Vector3d> &bezier_control_points, bool use_laplace_beltrami);

	std::vector<Eigen::Vector3d> &get_bezier_control_points() { return m_bezier_control_points; }
	const std::vector<Eigen::Vector3d> &get_bezier_control_points() const { return m_bezier_control_points; }

	const Eigen::SparseMatrix<double> &get_ind_to_full_matrix() { return m_ind2full; };
	const Eigen::SparseMatrix<double> &get_full_to_ind_matrix() { return m_full2ind; };
	const Eigen::SparseMatrix<double> &get_stiffness_matrix() { return m_stiffness_matrix; };
	const Eigen::SparseMatrix<double> &get_position_matrix() { return m_position_matrix; };
	const Eigen::MatrixXd &get_vertices() const { return m_V; }
	const Eigen::MatrixXi &get_faces() const { return m_F; }
	const AffineManifold &get_affine_manifold() const { return m_affine_manifold; }

	double fitting_weight;
    bool double_area = false;
    bool invert_area = false;
	/**
	 * @brief Assemble the stiffness matrix for the parameterization metric Laplacian energy
	 * in terms of Bezier coordinates.
	 *
	 * @return Laplacian energy stiffness matrix
	 */
	Eigen::SparseMatrix<double> generate_laplacian_stiffness_matrix() const;

	/**
	 * @brief Assemble the stiffness matrix for the surface metric Laplace Beltrami energy
	 * in terms of Bezier coordinates.
	 *
	 * @param bezier_control_points: list of 3D Bezier nodes
	 * @return Laplace Beltrami energy stiffness matrix
	 */
	Eigen::SparseMatrix<double> generate_laplace_beltrami_stiffness_matrix(const std::vector<Eigen::Vector3d> &bezier_control_points) const;

	double compute_normalized_fitting_weight() const;

	void initialize_data_log();
	void write_data_log_entry();
	void close_logs();

private:
	igl::Timer timer;
	igl::Timer total_timer;
	typedef Eigen::Matrix<double, 10, 10> CubicHessian;
	typedef Eigen::Triplet<double> Triplet;

	Eigen::SparseMatrix<double> m_stiffness_matrix;
	Eigen::SparseMatrix<double> m_position_matrix;

	std::vector<Eigen::Vector3d> m_bezier_control_points;

	Eigen::MatrixXd m_V;
	Eigen::MatrixXi m_F;
	AffineManifold m_affine_manifold;

	Eigen::SparseMatrix<double> m_full2ind, m_ind2full;
	
	std::string output_dir = "./";
  std::ofstream log_file;

	struct IterationData
	{
		int iter;
		double initial_energy;
		double optimized_energy;
		double step_size;
		double total_time;
		double assemble_time;
		double solve_time;
		double solve_residual;
		double constraint_error;
	};
	IterationData ID;

	/**
	 * @brief Helper function to produce the constraint and independent variable projection matrices.
	 *
	 */
	void initialize_ind_to_full_matrices();

	/**
	 * @brief Get the Bezier node indices of the three micro-triangles of a given face.
	 *
	 * in order: 003 300 030 102 201 210 120 021 012 111
	 * TODO: taken from clough_tocher_surface.cpp; make a common reference
	 *
	 * @param face_index: mesh face index
	 * @return array of 3 patch Bezier nodes (10 node indices per patch)
	 */
	std::array<std::array<int64_t, 10>, 3> get_micro_triangle_nodes(int64_t face_index) const;

	std::array<std::array<int64_t, 10>, 3> get_local_micro_triangle_nodes() const
	{
	  return {{{{0, 1, 18, 3, 4, 14, 15, 13, 12, 9}},
			{{1, 2, 18, 5, 6, 16, 17, 15, 14, 10}},
			{{2, 0, 18, 7, 8, 12, 13, 17, 16, 11}}}};
	}

	/**
	 * @brief Assemble the matrix to extract the vertex position nodes from the Bezier node vector.
	 *
	 * @return node position matrix
	 */
	Eigen::SparseMatrix<double> generate_position_matrix() const;

	// *****************
	// Utility Functions
	// *****************

	/**
	 * @brief Helper function to evaluate a quadratic energy 0.5 x^T H x + d^T x + E0
	 *
	 * @param H: quadratic energy Hessian
	 * @param d: quadratic energy derivative
	 * @param E0: quadratic energy constant term
	 * @param x: variable
	 * @return energy value
	 */
	double evaluate_quadratic_energy(
			const Eigen::SparseMatrix<double> &H,
			const Eigen::VectorXd &d,
			const double &E0,
			const Eigen::VectorXd &x);

	/**
	 * @brief Given a square matrix mat, produce the kronecker product matrix mat (x) I_3.
	 *
	 * @param mat: matrix to triple
	 * @return tripled matrix
	 */
	Eigen::SparseMatrix<double> triple_matrix(const Eigen::SparseMatrix<double> &mat) const;

	/**
	 * @brief Given a list of global Bezier nodes, generate the vector of full Bezier variables.
	 *
	 * The order is [x0, y0, z0, x1, ...]
	 *
	 * @param bezier_control_points: list of 3D Bezier nodes
	 * @return flattened variable vector
	 */
	Eigen::VectorXd build_node_vector(const std::vector<Eigen::Vector3d> &bezier_control_points);

	/**
	 * @brief Given a vector of full Bezier variables, construct a list of Bezier nodes
	 *
	 * The assumed order is [x0, y0, z0, x1, ...]
	 *
	 * @param p: flattened varibale vector
	 * @return list of 3D variable nodes
	 */
	std::vector<Eigen::Vector3d> build_control_points(const Eigen::VectorXd &p);

	/**
	 * @brief Helper function to assemble to local stiffness matrix for a given face into the global
	 * laplacian stiffness matrix.
	 *
	 * @param face_uv_positions: uv coordiantes of the face vertices
	 * @param patch_indices: Bezier node indices of the three micro-triangle patches of the face
	 * @param stiffness_matrix_trips: IJV triplets for the global stiffness matrix
	 */
	void assemble_local_laplacian_siffness_matrix(
			const std::array<PlanarPoint, 3> &face_uv_positions,
			const std::array<std::array<int64_t, 10>, 3> &patch_indices,
			std::vector<Triplet> &stiffness_matrix_trips) const;

	/**
	 * @brief Helper function to assemble to local stiffness matrix for a given face into the global
	 * laplace beltrami stiffness matrix.
	 *
	 * @param bezier_control_points: list of 3D Bezier nodes
	 * @param patch_indices: Bezier node indices of the three micro-triangle patches of the face
	 * @param stiffness_matrix_trips: IJV triplets for the global stiffness matrix
	 */
	void assemble_local_laplace_beltrami_siffness_matrix(
			const std::vector<Eigen::Vector3d> &bezier_control_points,
			const std::array<std::array<int64_t, 10>, 3> &patch_indices,
			std::vector<Triplet> &stiffness_matrix_trips) const;

	/**
	 * @brief Helper function to assemble local hessian into the global Hessian.
	 *
	 * @param patch_indices: mapping from local to global indices
	 * @param local_hessian: local Hessian
	 * @param global_hessian_trips: global Hessian (represented with IJV triplets)
	 */
	void assemble_patch_coefficients(
			const std::array<int64_t, 10> &patch_indices,
			const CubicHessian &local_hessian,
			std::vector<Triplet> &global_hessian_trips);
};

std::vector<Eigen::Vector3d>
generate_linear_clough_tocher_surface(CloughTocherSurface &ct_surface,
                                      const Eigen::MatrixXd &V);

void set_bezier_control_points(CloughTocherSurface &ct_surface,
                const std::vector<Eigen::Vector3d> &bezier_control_points);

// Helper function to write a curface with external bezier nodes to file
void write_mesh(CloughTocherSurface &ct_surface,
                const std::vector<Eigen::Vector3d> &bezier_control_points,
                const std::string &filename);

// write edge geometry to file
void write_polylines_to_obj(
    const std::string& filename,
    const std::vector<SpatialVector>& points,
    const std::vector<std::vector<int>>& polylines
);
