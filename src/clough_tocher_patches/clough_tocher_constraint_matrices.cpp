#include "clough_tocher_constraint_matrices.hpp"
#include "clough_tocher_autogen_constraint_matrices.hpp"

Eigen::Matrix<double, 12, 12> L_L2d_ind_m() {
  double L[12][12];
  L_L2d_ind_matrix(L);
  Eigen::Matrix<double, 12, 12> L_L2d_ind_mat;

  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 12; ++j) {
      L_L2d_ind_mat(i, j) = L[i][j];
    }
  }

  return L_L2d_ind_mat;
}

Eigen::Matrix<double, 7, 12> L_d2L_dep_m() {
  double L[7][12];
  L_d2L_dep_matrix(L);
  Eigen::Matrix<double, 7, 12> L_d2L_dep_mat;

  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 12; ++j) {
      L_d2L_dep_mat(i, j) = L[i][j];
    }
  }

  return L_d2L_dep_mat;
}

Eigen::Matrix<double, 5, 1> c_e_m() {
  double c[5];
  c_e_matrix(c);
  Eigen::Matrix<double, 5, 1> c_e_mat;

  for (int i = 0; i < 5; ++i) {
    c_e_mat[i] = c[i];
  }

  return c_e_mat;
}

std::array<std::array<Eigen::Matrix<double, 1, 12>, 3>, 3> c_t_m() {
  double c[3][3][12];
  c_t_matrix(c);
  std::array<std::array<Eigen::Matrix<double, 1, 12>, 3>, 3> c_t_mat;
  for (int j = 0; j < 3; ++j) {
    for (int m = 0; m < 3; ++m) {
      for (int i = 0; i < 12; ++i) {
        c_t_mat[j][m][i] = c[j][m][i];
      }
    }
  }

  return c_t_mat;
}

// bezier
Eigen::Matrix<double, 7, 1> c_hij_m() {
  double c[7];
  c_hij_matrix(c);
  Eigen::Matrix<double, 7, 1> c_hij_mat;

  for (int i = 0; i < 7; ++i) {
    c_hij_mat[i] = c[i];
  }

  return c_hij_mat;
}

Eigen::Matrix<double, 10, 10> p3_lag2bezier_m() {
  double c[10][10];
  p3_lag2bezier_matrix(c);

  Eigen::Matrix<double, 10, 10> p3_lag2bezier_mat;

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      p3_lag2bezier_mat(i, j) = c[i][j];
    }
  }

  return p3_lag2bezier_mat;
}