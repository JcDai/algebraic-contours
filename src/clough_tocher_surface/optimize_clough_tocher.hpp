#pragma once

#include "affine_manifold.h"
#include "clough_tocher_surface.hpp"
#include "common.h"

// TODO: Move to cpp file
#include <Eigen/CholmodSupport>
#include <Eigen/Sparse>
#include <igl/Timer.h>
#include <igl/per_vertex_normals.h>
#include <unsupported/Eigen/SparseExtra>

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
  CloughTocherOptimizer(const Eigen::MatrixXd V,
                        const Eigen::MatrixXi F,
                        const AffineManifold affine_manifold,
                        bool use_incenter = false,
                        bool skip_cone_constraints = false);

  /**
   * @brief Optimize the quadratic Laplacian energy over the parameterization
   * metric with fitting term while maintining the C1 constraints.
   *
   * @param bezier_control_points: initial Bezier points (including positions
   * for fitting term)
   * @return optimized control points
   */
  std::vector<Eigen::Vector3d> optimize_laplacian_energy(
      const std::vector<Eigen::Vector3d>& bezier_control_points);

  /**
   * @brief Optimize the quadratic Laplacian energy over the parameterization
   * metric with fitting term by **TRACKED VERTICES** while maintining the C1
   * constraints.
   *
   * @param bezier_control_points: initial Bezier points
   * @return optimized control points
   */
  std::vector<Eigen::Vector3d> optimize_laplacian_energy_tracked(
      const std::vector<Eigen::Vector3d>& bezier_control_points);

  /**
   * @brief Optimize the Laplace Beltrami energy with fitting term, starting
   * over the inital surface metric, while maintining the C1 constraints.
   *
   * @param bezier_control_points: initial Bezier points (including positions
   * for fitting term)
   * @param iterations: number of iterations of metric optimization to apply
   * @return optimized control points
   */
  std::vector<Eigen::Vector3d> optimize_laplace_beltrami_energy(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      int iterations = 1,
      double step_size = 1.);

  std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
  generate_position_energy_quadratic_tracked(
      const Eigen::VectorXd& v0,
      const Eigen::SparseMatrix<double>& A,
      const std::vector<Eigen::Vector3d>& optimized_control_points) const;

  std::vector<Eigen::Vector3d> gradient_descent_laplace_beltrami_energy(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      int iterations = 10,
      double step_size = 1e-4);

  /**
   * @brief Optimize the Laplace Beltrami energy with fitting term by **TRACKED
   * VERTICES** , starting over the inital surface metric, while maintining the
   * C1 constraints.
   *
   * @param bezier_control_points: initial Bezier points (including positions
   * for fitting term)
   * @param iterations: number of iterations of metric optimization to apply
   * @return optimized control points
   */
  std::vector<Eigen::Vector3d> optimize_laplace_beltrami_energy_tracked(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      int iterations = 1,
      double step_size = 1.);

  // TODO:
  // std::vector<Eigen::Vector3d>
  // gradient_descent_laplace_beltrami_energy_tracked(
  //     const std::vector<Eigen::Vector3d>& bezier_control_points,
  //     int iterations = 10,
  //     double step_size = 1e-4);

  /**
   * @brief Project the surface determined by the control points to the
   * constraint.
   *
   * @param bezier_control_points: initial Bezier control points
   * @return projected Bezier control points
   */
  std::vector<Eigen::Vector3d> project_to_constraints(
      const std::vector<Eigen::Vector3d>& bezier_control_points) const;

  /**
   * @brief Evaluate the quadratic Laplacian energy over the parameterization
   * metric with fitting term.
   *
   * @param bezier_control_points: Bezier points
   * @return energy for the Bezier points
   */
  double evaluate_energy(
      const std::vector<Eigen::Vector3d>& bezier_control_points);

  Eigen::SparseMatrix<double> generate_laplace_beltrami_stiffness_matrix()
      const;

  std::vector<double> compute_face_energies(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      bool use_laplace_beltrami);

  std::vector<Eigen::Vector3d>& get_bezier_control_points()
  {
    return m_bezier_control_points;
  }
  const std::vector<Eigen::Vector3d>& get_bezier_control_points() const
  {
    return m_bezier_control_points;
  }

  const Eigen::SparseMatrix<double>& get_ind_to_full_matrix() const
  {
    return m_ind2full;
  };
  const Eigen::SparseMatrix<double>& get_full_to_ind_matrix() const
  {
    return m_full2ind;
  };
  const Eigen::SparseMatrix<double>& get_stiffness_matrix() const
  {
    return m_stiffness_matrix;
  };
  const Eigen::SparseMatrix<double>& get_position_matrix() const
  {
    return m_position_matrix;
  };
  const Eigen::MatrixXd& get_vertices() const { return m_V; }
  const Eigen::MatrixXi& get_faces() const { return m_F; }
  const AffineManifold& get_affine_manifold() const
  {
    return m_affine_manifold;
  }

  double fitting_weight;
  bool double_area = false;
  bool invert_area = false;
  bool normalize_count = false;
  bool bound_energy =
      true; // bound the energy during Laplace Beltrami optimization
  bool bound_residual =
      true; // bound the residual during Laplace Beltrami optimization
  bool use_orthogonal_projection =
      true; // use orthongonal projection to constraint subset
  bool use_parametric_metric = false; // use parameterization metric for first
                                      // iteration of Laplace Beltrami
  bool use_fixed_metric = false; // use fixed metric for gradient computation

  bool m_skip_cone_constraints = false;

  /**
   * @brief Assemble the stiffness matrix for the parameterization metric
   * Laplacian energy in terms of Bezier coordinates.
   *
   * @return Laplacian energy stiffness matrix
   */
  Eigen::SparseMatrix<double> generate_laplacian_stiffness_matrix() const;

  /**
   * @brief Assemble the stiffness matrix for the surface metric Laplace
   * Beltrami energy in terms of Bezier coordinates.
   *
   * @param bezier_control_points: list of 3D Bezier nodes
   * @return Laplace Beltrami energy stiffness matrix
   */
  Eigen::SparseMatrix<double> generate_laplace_beltrami_stiffness_matrix(
      const std::vector<Eigen::Vector3d>& bezier_control_points) const;

  double compute_normalized_fitting_weight() const;
  double compute_normalized_fitting_weight_tracked() const;

  Eigen::SparseMatrix<double> generate_position_matrix(
      const Eigen::VectorXd& p) const;

  void initialize_data_log();
  void write_data_log_entry();
  void close_logs();
  void checkpoint_control_points(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      int iter);

  std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
  generate_position_energy_quadratic(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      const std::vector<Eigen::Vector3d>& optimized_control_points) const;

  std::string output_dir = "./";
  int p_norm = 2;

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
  CloughTocherSurface ct_surface;

  Eigen::SparseMatrix<double> m_full2ind, m_ind2full;

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

  bool m_use_incenter;

  /**
   * @brief Helper function to produce the constraint and independent variable
   * projection matrices.
   *
   */
  void initialize_ind_to_full_matrices(bool use_incenter = false);

  /**
   * @brief Get the Bezier node indices of the three micro-triangles of a given
   * face.
   *
   * in order: 003 300 030 102 201 210 120 021 012 111
   * TODO: taken from clough_tocher_surface.cpp; make a common reference
   *
   * @param face_index: mesh face index
   * @return array of 3 patch Bezier nodes (10 node indices per patch)
   */
  std::array<std::array<int64_t, 10>, 3> get_micro_triangle_nodes(
      int64_t face_index) const;

  std::array<std::array<int64_t, 10>, 3> get_local_micro_triangle_nodes() const
  {
    return { { { { 0, 1, 18, 3, 4, 14, 15, 13, 12, 9 } },
               { { 1, 2, 18, 5, 6, 16, 17, 15, 14, 10 } },
               { { 2, 0, 18, 7, 8, 12, 13, 17, 16, 11 } } } };
  }

  /**
   * @brief Assemble the matrix to extract the vertex position nodes from the
   * Bezier node vector.
   *
   * @return node position matrix
   */
  Eigen::SparseMatrix<double> generate_position_matrix() const;

  // *****************
  // Utility Functions
  // *****************

  /**
   * @brief Helper function to evaluate a quadratic energy 0.5 x^T H x + d^T x +
   * E0
   *
   * @param H: quadratic energy Hessian
   * @param d: quadratic energy derivative
   * @param E0: quadratic energy constant term
   * @param x: variable
   * @return energy value
   */
  double evaluate_quadratic_energy(const Eigen::SparseMatrix<double>& H,
                                   const Eigen::VectorXd& d,
                                   const double& E0,
                                   const Eigen::VectorXd& x);

  /**
   * @brief Given a square matrix mat, produce the kronecker product matrix mat
   * (x) I_3.
   *
   * @param mat: matrix to triple
   * @return tripled matrix
   */
  Eigen::SparseMatrix<double> triple_matrix(
      const Eigen::SparseMatrix<double>& mat) const;

  /**
   * @brief Given a list of global Bezier nodes, generate the vector of full
   * Bezier variables.
   *
   * The order is [x0, y0, z0, x1, ...]
   *
   * @param bezier_control_points: list of 3D Bezier nodes
   * @return flattened variable vector
   */
  Eigen::VectorXd build_node_vector(
      const std::vector<Eigen::Vector3d>& bezier_control_points) const;

  /**
   * @brief Given a vector of full Bezier variables, construct a list of Bezier
   * nodes
   *
   * The assumed order is [x0, y0, z0, x1, ...]
   *
   * @param p: flattened varibale vector
   * @return list of 3D variable nodes
   */
  std::vector<Eigen::Vector3d> build_control_points(
      const Eigen::VectorXd& p) const;

  /**
   * @brief Project a vector orthogonally to the reduced constraint subspace.
   *
   * @param p0: initial Bezier node vector
   * @return reduced space vector
   */
  Eigen::VectorXd project_to_reduced_subspace(const Eigen::VectorXd& p0) const;

  /**
   * @brief Helper function to assemble to local stiffness matrix for a given
   * face into the global laplacian stiffness matrix.
   *
   * @param face_uv_positions: uv coordiantes of the face vertices
   * @param patch_indices: Bezier node indices of the three micro-triangle
   * patches of the face
   * @param stiffness_matrix_trips: IJV triplets for the global stiffness matrix
   */
  void assemble_local_laplacian_siffness_matrix(
      const std::array<PlanarPoint, 3>& face_uv_positions,
      const std::array<std::array<int64_t, 10>, 3>& patch_indices,
      std::vector<Triplet>& stiffness_matrix_trips) const;

  /**
   * @brief Helper function to assemble to local stiffness matrix for a given
   * face into the global laplace beltrami stiffness matrix.
   *
   * @param bezier_control_points: list of 3D Bezier nodes
   * @param patch_indices: Bezier node indices of the three micro-triangle
   * patches of the face
   * @param stiffness_matrix_trips: IJV triplets for the global stiffness matrix
   */
  void assemble_local_laplace_beltrami_siffness_matrix(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      const std::array<std::array<int64_t, 10>, 3>& patch_indices,
      std::vector<Triplet>& stiffness_matrix_trips) const;

  std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
  generate_autodiff_laplace_beltrami_stiffness_matrix(
      const std::vector<Eigen::Vector3d>& bezier_control_points) const;

  void assemble_autodiff_laplace_beltrami_siffness_matrix(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      const std::array<std::array<int64_t, 10>, 3>& patch_indices,
      double& energy,
      Eigen::VectorXd& gradient,
      std::vector<Triplet>& stiffness_matrix_trips) const;

  std::tuple<double, Eigen::VectorXd>
  generate_autodiff_laplace_beltrami_gradient(
      const std::vector<Eigen::Vector3d>& bezier_control_points) const;

  void assemble_autodiff_laplace_beltrami_gradient(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      const std::array<std::array<int64_t, 10>, 3>& patch_indices,
      double& energy,
      Eigen::VectorXd& gradient) const;

  /**
   * @brief Helper function to assemble local hessian into the global Hessian.
   *
   * @param patch_indices: mapping from local to global indices
   * @param local_hessian: local Hessian
   * @param global_hessian_trips: global Hessian (represented with IJV triplets)
   */
  void assemble_patch_coefficients(const std::array<int64_t, 10>& patch_indices,
                                   const CubicHessian& local_hessian,
                                   std::vector<Triplet>& global_hessian_trips);

  // for tracked vertices
public:
  struct TrackedVertex
  {
    int64_t macro_tri_id = -1;

    Eigen::Vector3d pos_3d;
    Eigen::Vector2d local_uv_pos;
    double one_ring_area;

    Eigen::Vector3d dfdu;
    Eigen::Vector3d dfdv;

    Eigen::Vector2d old_v0;
    Eigen::Vector2d old_v1;
    Eigen::Vector2d old_v2;

    Eigen::Vector3d normal;
  };

  bool fit_tracked_vertices;

  std::vector<TrackedVertex> m_tracked_vertices;

  /**
   * @brief get face node matrix for tracked vertices of size [19 *
   * #tracked_vertices, #control_points]
   */
  void P_G2F_tracked(Eigen::SparseMatrix<double>& m);

  /**
   * @brief get 19 to 10 nodes block diagonal matrix. size [10 * 19] *
   * #tracked_vertices at diagonal
   */
  void Macro2Micro_tracked(Eigen::SparseMatrix<double>& m);

  /**
   * @brief get (bezier coeff * uvw values) matrix, size [1 * 10] * #
   * tracked_vertices block diagonal
   */
  void bezier_coeff_with_uv_value_tracked(Eigen::SparseMatrix<double>& m);

  void dudv_bezier_coeff_with_uv_value_tracked(
      Eigen::SparseMatrix<double>& m_u,
      Eigen::SparseMatrix<double>& m_v);

  void compute_area_weighted_fitting_weight_matrix_tracked(
      Eigen::SparseMatrix<double>& m);

  std::array<Eigen::Vector3d, 2> transform_tangent_old_to_new(
      const Eigen::Vector2d& old_v0,
      const Eigen::Vector2d& old_v1,
      const Eigen::Vector2d& old_v2,
      const Eigen::Vector2d& new_v0,
      const Eigen::Vector2d& new_v1,
      const Eigen::Vector2d& new_v2,
      const Eigen::Vector3d& old_du,
      const Eigen::Vector3d& old_dv);

  void compute_tracked_vertices_normals();

  /**
   * @brief compute N from N * A_tangent * p, size of (#tracked_vertice by 3 *
   * #tracked_vertice), block [1, 3] diagonal
   */
  void normal_matrix_tracked(Eigen::SparseMatrix<double>& m);

  void sqrt_area_weight_tracked(Eigen::SparseMatrix<double>& m);
  void sqrt_area_weight_tracked_triple(Eigen::SparseMatrix<double>& m);

  /**
   * @brief generate position matrix P for tracked vertices.
   * size [10 * #tracked, #bezier_cp]
   * P * N  = traced 3d pos in cubic triangles. replace
   * generate_position_matrix()
   */
  Eigen::SparseMatrix<double> generate_tracked_position_matrix();

  /**
   * @brief generate weight * A_pos stack (1 - weight) * A_normal (n_target dot
   * du + n_target dot dv)
   */
  Eigen::SparseMatrix<double> generate_tracked_position_normal_matrix(
      double weight);

  Eigen::VectorXd build_tracked_vertices_vector();
  Eigen::VectorXd build_tracked_vertices_pos_normal_vector(double w);

  std::vector<Eigen::Vector3d> evaluate_tracked_vertices(
      const std::vector<Eigen::Vector3d>& bezier_control_points);

  // test tracked vertices
  std::vector<Eigen::Vector3d> optimize_fitting_term_direct(
      const std::vector<Eigen::Vector3d>& bezier_control_points);

  std::vector<Eigen::Vector3d> optimize_fitting_pos_and_normal_without_c1(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      double weight);

  std::vector<Eigen::Vector3d> optimize_fitting_term_iterative(
      const std::vector<Eigen::Vector3d>& bezier_control_points,
      int iterations,
      double step_size);

  std::vector<Eigen::Vector3d> direct_fitting_without_c1(
      const std::vector<Eigen::Vector3d>& bezier_control_points);

  void serialize_dofs(
      const std::string& filename,
      CloughTocherSurface& ct_surface,
      const std::vector<Eigen::Vector3d>& bezier_control_points);
};

std::vector<Eigen::Vector3d>
generate_linear_clough_tocher_surface(CloughTocherSurface& ct_surface,
                                      const Eigen::MatrixXd& V);

void
set_bezier_control_points(
    CloughTocherSurface& ct_surface,
    const std::vector<Eigen::Vector3d>& bezier_control_points);

// Helper function to write a curface with external bezier nodes to file
void
write_mesh(CloughTocherSurface& ct_surface,
           const std::vector<Eigen::Vector3d>& bezier_control_points,
           const std::string& filename);

void
write_tracked_vertices(
    CloughTocherSurface& ct_surface,
    const std::vector<Eigen::Vector3d>& bezier_control_points,
    const std::string& filename);

void
write_tracked_vertices_with_subdivision_level(
    CloughTocherSurface& ct_surface,
    const std::vector<Eigen::Vector3d>& bezier_control_points,
    const std::string& filename,
    int subdivision_level);

void
write_full_tracked_vertices_with_subdivision_level(
    CloughTocherSurface& ct_surface,
    const std::vector<Eigen::Vector3d>& bezier_control_points,
    const std::string& filename,
    int subdivision_level,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

// write edge geometry to file
void
write_polylines_to_obj(const std::string& filename,
                       const std::vector<SpatialVector>& points,
                       const std::vector<std::vector<int>>& polylines);

bool
compute_newton_update_dir_with_reg(Eigen::SparseMatrix<double>& hessian,
                                   Eigen::VectorXd& derivative,
                                   Eigen::VectorXd& x,
                                   double initial_reg_weight = 1.,
                                   double reg_weight_inc = 10.,
                                   double max_reg_weight = 1e8);
