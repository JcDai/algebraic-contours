#include "clough_tocher_patch.hpp"
#include "clough_tocher_matrices.hpp"

#include <fstream>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>

const std::array<Eigen::Matrix<double, 3, 3>, 3>
    CloughTocherPatch::m_CTtri_bounds = CT_subtri_bound_matrices();

const std::array<Eigen::Matrix<double, 10, 12>, 3>
    CloughTocherPatch::m_CT_matrices = CT_subtri_matrices();

CloughTocherPatch::CloughTocherPatch(
    Eigen::Matrix<double, 12, 3> &boundary_data)
    : m_boundary_data(boundary_data) {

  // TODO: old deprecated code, hij are quadratic, wrong , just here for
  // reference
  // m_boundary_data.row(0) = m_corner_data[0].function_value; // p0
  // m_boundary_data.row(1) = m_corner_data[1].function_value; // p1
  // m_boundary_data.row(2) = m_corner_data[2].function_value; // p2

  // m_boundary_data.row(3) = m_corner_data[0].first_edge_derivative;  // d01
  // m_boundary_data.row(4) = m_corner_data[1].second_edge_derivative; // d10
  // m_boundary_data.row(5) = m_corner_data[1].first_edge_derivative;  // d12
  // m_boundary_data.row(6) = m_corner_data[2].second_edge_derivative; // d21
  // m_boundary_data.row(7) = m_corner_data[2].first_edge_derivative;  // d20
  // m_boundary_data.row(8) = m_corner_data[0].second_edge_derivative; // d02

  // m_boundary_data.row(9) = m_midpoint_data[2].normal_derivative;  // h01
  // m_boundary_data.row(10) = m_midpoint_data[0].normal_derivative; // h12
  // m_boundary_data.row(11) = m_midpoint_data[1].normal_derivative; // h20

  // compute coeff matrices
  for (int i = 0; i < 3; ++i) {
    m_CT_coeffs[i] = m_CT_matrices[i] * m_boundary_data;
  }
}

int CloughTocherPatch::triangle_ind(const double &u, const double &v,
                                    const double &w) const {
  int idx = -1;
  for (int i = 0; i < 3; ++i) {
    if (m_CTtri_bounds[i](0, 0) * u + m_CTtri_bounds[i](0, 1) * v +
                m_CTtri_bounds[i](0, 2) * w >=
            -1e-7 &&
        m_CTtri_bounds[i](1, 0) * u + m_CTtri_bounds[i](1, 1) * v +
                m_CTtri_bounds[i](1, 2) * w >=
            -1e-7 &&
        m_CTtri_bounds[i](2, 0) * u + m_CTtri_bounds[i](2, 1) * v +
                m_CTtri_bounds[i](2, 2) * w >=
            -1e-7) {
      idx = i;
      break;
    }
  }

  assert(idx > -1);
  return idx;
}

Eigen::Matrix<double, 10, 1>
CloughTocherPatch::monomial_basis_eval(const double &u, const double &v,
                                       const double &w) const {
  Eigen::Matrix<double, 10, 1> monomial_basis_values;
  monomial_basis_values(0, 0) = w * w * w; // w3
  monomial_basis_values(1, 0) = v * w * w; // vw2
  monomial_basis_values(2, 0) = v * v * w; // v2w
  monomial_basis_values(3, 0) = v * v * v; // v3
  monomial_basis_values(4, 0) = u * w * w; // uw2
  monomial_basis_values(5, 0) = u * v * w; // uvw
  monomial_basis_values(6, 0) = u * v * v; // uv2
  monomial_basis_values(7, 0) = u * u * w; // u2w
  monomial_basis_values(8, 0) = u * u * v; // u2v
  monomial_basis_values(9, 0) = u * u * u; // u3

  return monomial_basis_values;
}

Eigen::Matrix<double, 3, 1> CloughTocherPatch::CT_eval(const double &u,
                                                       const double &v) const {
  const double w = 1.0 - u - v;
  int idx = CloughTocherPatch::triangle_ind(u, v, w);

  // std::cout << "subtri_idx: " << idx << std::endl;
  Eigen::Matrix<double, 10, 1> bb_vector =
      CloughTocherPatch::monomial_basis_eval(u, v, w);

  // std::cout << "monomial: " << bb_vector << std::endl;

  Eigen::Matrix<double, 3, 1> val;
  val = m_CT_coeffs[idx].transpose() * bb_vector;
  return val;
}

std::array<Eigen::Matrix<double, 10, 3>, 3>
CloughTocherPatch::get_coeffs() const {
  return m_CT_coeffs;
}

double CloughTocherPatch::external_boundary_data_eval(
    const double &u, const double &v,
    Eigen::Matrix<double, 12, 1> &external_boundary_data) const {
  const double w = 1.0 - u - v;
  int idx = CloughTocherPatch::triangle_ind(u, v, w);

  Eigen::Matrix<double, 10, 1> bb_vector =
      CloughTocherPatch::monomial_basis_eval(u, v, w);

  double value =
      (m_CT_matrices[idx] * external_boundary_data).transpose() * bb_vector;

  return value;
}