#pragma once

#include "clough_tocher_patch.hpp"
#include "common.h"
#include "optimize_spline_surface.h"
#include "position_data.h"

class CloughTocherSurface {
public:
  typedef size_t PatchIndex;

  CloughTocherSurface();
  CloughTocherSurface(const Eigen::MatrixXd &V,
                      const AffineManifold &affine_manifold,
                      const OptimizationParameters &optimization_params,
                      Eigen::SparseMatrix<double> &fit_matrix,
                      Eigen::SparseMatrix<double> &energy_hessian,
                      Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
                          &energy_hessian_inverse);

  Eigen::Matrix<double, 1, 3> evaluate_patch(const PatchIndex &patch_index,
                                             const double &u, const double &v);

  void write_cubic_surface_to_msh_no_conn(std::string filename);
  void write_coeffs_to_obj(std::string filename);
  void sample_to_obj(std::string filename, int sample_size = 10);

  void write_cubic_surface_to_msh_with_conn(std::string filename);
  void write_cubic_surface_to_msh_with_conn_from_lagrange_nodes(
      std::string filename, bool write_bezier = false);
  void write_external_bd_interpolated_function_values_from_lagrange_nodes(
      std::string filename,
      std::vector<Eigen::Matrix<double, 12, 1>> &external_boundary_data);

  // for bilaplacian
  void write_connected_lagrange_nodes(std::string filename, Eigen::MatrixXd &V);
  void write_connected_lagrange_nodes_values(std::string filename);

public:
  void generate_face_normals(const Eigen::MatrixXd &V,
                             const AffineManifold &affine_manifold,
                             Eigen::MatrixXd &N);

public:
  std::vector<CloughTocherPatch> m_patches;

  AffineManifold m_affine_manifold;
  std::vector<std::array<TriangleCornerFunctionData, 3>> m_corner_data;
  std::vector<std::array<TriangleMidpointFunctionData, 3>> m_midpoint_data;

  std::vector<Eigen::Vector3d> m_lagrange_node_values;
  std::vector<Eigen::Vector3d> m_bezier_control_points;

public:
  // constraint matrices

  // interior constraints
  void P_G2F(Eigen::SparseMatrix<double> &m);
  void C_L_int(Eigen::Matrix<double, 7, 19> &m);
  void C_F_int(Eigen::SparseMatrix<double> &m);

  // edge endpoint constraints
  void P_G2E(Eigen::SparseMatrix<double> &m);
  void C_E_end(Eigen::SparseMatrix<double> &m,
               Eigen::SparseMatrix<double> &m_elim);

  // edge midpoint constraints
  void C_E_mid(Eigen::SparseMatrix<double> &m);

  // cone constraints
  void diag_P_G2F(Eigen::SparseMatrix<double> &m); // diag(p_g2f, p_g2f, p_g2f)
  void P_3D(Eigen::SparseMatrix<double> &m);       // group x,y,z values by face
  void C_F_cone(Eigen::SparseMatrix<double> &m, Eigen::MatrixXd &v_normals);

  // independent -> dependent matrices using bezier basis
  void Ci_endpoint_ind2dep(Eigen::SparseMatrix<double> &m,
                           std::vector<int64_t> &constrained_row_ids,
                           std::map<int64_t, int> &independent_node_map);
  void Ci_internal_ind2dep_1(Eigen::SparseMatrix<double> &m,
                             std::vector<int64_t> &constrained_row_ids,
                             std::map<int64_t, int> &independent_node_map);
  void Ci_midpoint_ind2dep(Eigen::SparseMatrix<double> &m,
                           std::vector<int64_t> &constrained_row_ids,
                           std::map<int64_t, int> &independent_node_map);
  void Ci_internal_ind2dep_2(Eigen::SparseMatrix<double> &m,
                             std::vector<int64_t> &constrained_row_ids,
                             std::map<int64_t, int> &independent_node_map);
  void Ci_cone_bezier(const Eigen::SparseMatrix<double> &m,
                      Eigen::SparseMatrix<double> &m_cone,
                      const Eigen::MatrixXd &v_normals);

  void bezier2lag_full_mat(Eigen::SparseMatrix<double, 1> &m);
  void lag2bezier_full_mat(Eigen::SparseMatrix<double, 1> &m);

  // beizer with cones
  void bezier_cone_constraints_expanded(Eigen::SparseMatrix<double, 1> &m,
                                        std::vector<int> &independent_node_map,
                                        std::vector<bool> &node_assigned,
                                        const Eigen::MatrixXd &v_normals);
  void bezier_endpoint_ind2dep_expanded(Eigen::SparseMatrix<double, 1> &m,
                                        std::vector<int> &independent_node_map,
                                        bool debug_isolate);
  void
  bezier_internal_ind2dep_1_expanded(Eigen::SparseMatrix<double, 1> &m,
                                     std::vector<int> &independent_node_map);
  void bezier_midpoint_ind2dep_expanded(Eigen::SparseMatrix<double, 1> &m,
                                        std::vector<int> &independent_node_map);
  void
  bezier_internal_ind2dep_2_expanded(Eigen::SparseMatrix<double, 1> &m,
                                     std::vector<int> &independent_node_map);

  void write_external_point_values_with_conn(const std::string &filename,
                                             const Eigen::MatrixXd &vertices);
};