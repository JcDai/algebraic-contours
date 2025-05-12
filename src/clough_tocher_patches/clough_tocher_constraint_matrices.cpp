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

Eigen::Matrix<double, 10, 10> p3_bezier2lag_m() {
  double c[10][10];
  p3_bezier2lag_matrix(c);

  Eigen::Matrix<double, 10, 10> p3_bezier2lag_mat;

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      p3_bezier2lag_mat(i, j) = c[i][j];
    }
  }

  return p3_bezier2lag_mat;
}

std::array<std::array<double, 13>, 8>
pi_cone_x_max_deps(const std::array<double, 3> &ac,
                   const std::array<double, 6> &am, const Eigen::Vector3d &ni) {

  std::array<std::array<double, 13>, 8> res;

  const double nix = ni[0];
  const double niy = ni[1];
  const double niz = ni[2];

  const double a1c = ac[0];
  const double a2c = ac[1];
  const double a3c = ac[2];

  const double a1m = am[0];
  const double a2m = am[1];
  const double a3m = am[2];
  const double a4m = am[3];
  const double a5m = am[4];
  const double a6m = am[5];

  // xji
  res[0] = {{1, niy / nix, niz / nix, 0, 0, 0, -niy / nix, -niz / nix, 0, 0, 0,
             0, 0}};
  // xmij
  res[1] = {{1, niy / nix, niz / nix, 0, 0, 0, 0, 0, 0, 0, 0, -niy / nix,
             -niz / nix}};
  // xmij_p
  res[2] = {{a1c * a5m + a1m + a3m + a6m,
             a1c * a5m * niy / nix + a3m * niy / nix + a6m * niy / nix,
             a1c * a5m * niz / nix + a3m * niz / nix + a6m * niz / nix,
             a2c * a5m + a2m, 0, 0, -a1c * a5m * niy / nix - a3m * niy / nix,
             -a1c * a5m * niz / nix - a3m * niz / nix, a3c * a5m + a4m, 0, 0,
             -a6m * niy / nix, -a6m * niz / nix}};
  // ymij_p
  res[3] = {{0, a1m, 0, 0, a2c * a5m + a2m, 0, a1c * a5m + a3m, 0, 0,
             a3c * a5m + a4m, 0, a6m, 0}};
  // zmij_p
  res[4] = {{0, 0, a1m, 0, 0, a2c * a5m + a2m, 0, a1c * a5m + a3m, 0, 0,
             a3c * a5m + a4m, 0, a6m}};
  // xcj_p
  res[5] = {{a1c, a1c * niy / nix, a1c * niz / nix, a2c, 0, 0, -a1c * niy / nix,
             -a1c * niz / nix, a3c, 0, 0, 0, 0}};
  // ycj_p
  res[6] = {{0, 0, 0, 0, a2c, 0, a1c, 0, 0, a3c, 0, 0, 0}};
  // zcj_p
  res[7] = {{0, 0, 0, 0, 0, a2c, 0, a1c, 0, 0, a3c, 0, 0}};

  return res;
}

std::array<std::array<double, 13>, 8>
pi_cone_y_max_deps(const std::array<double, 3> &ac,
                   const std::array<double, 6> &am, const Eigen::Vector3d &ni) {
  std::array<std::array<double, 13>, 8> res;

  const double nix = ni[0];
  const double niy = ni[1];
  const double niz = ni[2];

  const double a1c = ac[0];
  const double a2c = ac[1];
  const double a3c = ac[2];

  const double a1m = am[0];
  const double a2m = am[1];
  const double a3m = am[2];
  const double a4m = am[3];
  const double a5m = am[4];
  const double a6m = am[5];

  // yji
  res[0] = {{nix / niy, 1, niz / niy, 0, 0, 0, -nix / niy, -niz / niy, 0, 0, 0,
             0, 0}};
  // ymij
  res[1] = {{nix / niy, 1, niz / niy, 0, 0, 0, 0, 0, 0, 0, 0, -nix / niy,
             -niz / niy}};
  // xmij_p
  res[2] = {{a1m, 0, 0, a2c * a5m + a2m, 0, 0, a1c * a5m + a3m, 0,
             a3c * a5m + a4m, 0, 0, a6m, 0}};
  // ymij_p
  res[3] = {{a1c * a5m * nix / niy + a3m * nix / niy + a6m * nix / niy,
             a1c * a5m + a1m + a3m + a6m,
             a1c * a5m * niz / niy + a3m * niz / niy + a6m * niz / niy, 0,
             a2c * a5m + a2m, 0, -a1c * a5m * nix / niy - a3m * nix / niy,
             -a1c * a5m * niz / niy - a3m * niz / niy, 0, a3c * a5m + a4m, 0,
             -a6m * nix / niy, -a6m * niz / niy}};
  // zmij_p
  res[4] = {{0, 0, a1m, 0, 0, a2c * a5m + a2m, 0, a1c * a5m + a3m, 0, 0,
             a3c * a5m + a4m, 0, a6m}};
  // xcj_p
  res[5] = {{0, 0, 0, a2c, 0, 0, a1c, 0, a3c, 0, 0, 0, 0}};
  // ycj_p
  res[6] = {{a1c * nix / niy, a1c, a1c * niz / niy, 0, a2c, 0, -a1c * nix / niy,
             -a1c * niz / niy, 0, a3c, 0, 0, 0}};
  // zcj_p
  res[7] = {{0, 0, 0, 0, 0, a2c, 0, a1c, 0, 0, a3c, 0, 0}};

  return res;
}

std::array<std::array<double, 13>, 8>
pi_cone_z_max_deps(const std::array<double, 3> &ac,
                   const std::array<double, 6> &am, const Eigen::Vector3d &ni) {
  std::array<std::array<double, 13>, 8> res;

  const double nix = ni[0];
  const double niy = ni[1];
  const double niz = ni[2];

  const double a1c = ac[0];
  const double a2c = ac[1];
  const double a3c = ac[2];

  const double a1m = am[0];
  const double a2m = am[1];
  const double a3m = am[2];
  const double a4m = am[3];
  const double a5m = am[4];
  const double a6m = am[5];

  // zji
  res[0] = {{nix / niz, niy / niz, 1, 0, 0, 0, -nix / niz, -niy / niz, 0, 0, 0,
             0, 0}};
  // zmij
  res[1] = {{nix / niz, niy / niz, 1, 0, 0, 0, 0, 0, 0, 0, 0, -nix / niz,
             -niy / niz}};
  // xmij_p
  res[2] = {{a1m, 0, 0, a2c * a5m + a2m, 0, 0, a1c * a5m + a3m, 0,
             a3c * a5m + a4m, 0, 0, a6m, 0}};
  // ymij_p
  res[3] = {{0, a1m, 0, 0, a2c * a5m + a2m, 0, 0, a1c * a5m + a3m, 0,
             a3c * a5m + a4m, 0, 0, a6m}};
  // zmij_p
  res[4] = {{a1c * a5m * nix / niz + a3m * nix / niz + a6m * nix / niz,
             a1c * a5m * niy / niz + a3m * niy / niz + a6m * niy / niz,
             a1c * a5m + a1m + a3m + a6m, 0, 0, a2c * a5m + a2m,
             -a1c * a5m * nix / niz - a3m * nix / niz,
             -a1c * a5m * niy / niz - a3m * niy / niz, 0, 0, a3c * a5m + a4m,
             -a6m * nix / niz, -a6m * niy / niz}};
  // xcj_p
  res[5] = {{0, 0, 0, a2c, 0, 0, a1c, 0, a3c, 0, 0, 0, 0}};
  // ycj_p
  res[6] = {{0, 0, 0, 0, a2c, 0, 0, a1c, 0, a3c, 0, 0, 0}};
  // zcj_p
  res[7] = {{a1c * nix / niz, a1c * niy / niz, a1c, 0, 0, a2c, -a1c * nix / niz,
             -a1c * niy / niz, 0, 0, a3c, 0, 0}};

  return res;
}

std::array<std::array<double, 13>, 8>
pj_cone_x_max_deps(const std::array<double, 3> &ac,
                   const std::array<double, 6> &am, const Eigen::Vector3d &ni) {
  std::array<std::array<double, 13>, 8> res;

  const double nix = ni[0];
  const double niy = ni[1];
  const double niz = ni[2];

  const double a1c = ac[0];
  const double a2c = ac[1];
  const double a3c = ac[2];

  const double a1m = am[0];
  const double a2m = am[1];
  const double a3m = am[2];
  const double a4m = am[3];
  const double a5m = am[4];
  const double a6m = am[5];

  // xij
  res[0] = {{0, 0, 0, 1, niy / nix, niz / nix, -niy / nix, -niz / nix, 0, 0, 0,
             0, 0}};
  // xmij
  res[1] = {{0, 0, 0, 1, niy / nix, niz / nix, 0, 0, 0, 0, 0, -niy / nix,
             -niz / nix}};
  // xmij_p
  res[2] = {{a1m + a2c * a5m, 0, 0, a1c * a5m + a2m + a3m + a6m,
             a1c * a5m * niy / nix + a3m * niy / nix + a6m * niy / nix,
             a1c * a5m * niz / nix + a3m * niz / nix + a6m * niz / nix,
             -a1c * a5m * niy / nix - a3m * niy / nix,
             -a1c * a5m * niz / nix - a3m * niz / nix, a3c * a5m + a4m, 0, 0,
             -a6m * niy / nix, -a6m * niz / nix}};
  // ymij_p
  res[3] = {{0, a1m + a2c * a5m, 0, 0, a2m, 0, a1c * a5m + a3m, 0, 0,
             a3c * a5m + a4m, 0, a6m, 0}};
  // zmij_p
  res[4] = {{0, 0, a1m + a2c * a5m, 0, 0, a2m, 0, a1c * a5m + a3m, 0, 0,
             a3c * a5m + a4m, 0, a6m}};
  // xci_p
  res[5] = {{a2c, 0, 0, a1c, a1c * niy / nix, a1c * niz / nix, -a1c * niy / nix,
             -a1c * niz / nix, a3c, 0, 0, 0, 0}};
  // yci_p
  res[6] = {{0, a2c, 0, 0, 0, 0, a1c, 0, 0, a3c, 0, 0, 0}};
  // zci_p
  res[7] = {{0, 0, a2c, 0, 0, 0, 0, a1c, 0, 0, a3c, 0, 0}};

  return res;
}

std::array<std::array<double, 13>, 8>
pj_cone_y_max_deps(const std::array<double, 3> &ac,
                   const std::array<double, 6> &am, const Eigen::Vector3d &ni) {
  std::array<std::array<double, 13>, 8> res;

  const double nix = ni[0];
  const double niy = ni[1];
  const double niz = ni[2];

  const double a1c = ac[0];
  const double a2c = ac[1];
  const double a3c = ac[2];

  const double a1m = am[0];
  const double a2m = am[1];
  const double a3m = am[2];
  const double a4m = am[3];
  const double a5m = am[4];
  const double a6m = am[5];

  // yij
  res[0] = {{0, 0, 0, nix / niy, 1, niz / niy, -nix / niy, -niz / niy, 0, 0, 0,
             0, 0}};
  // ymij
  res[1] = {{0, 0, 0, nix / niy, 1, niz / niy, 0, 0, 0, 0, 0, -nix / niy,
             -niz / niy}};
  // xmij_p
  res[2] = {{a1m + a2c * a5m, 0, 0, a2m, 0, 0, a1c * a5m + a3m, 0,
             a3c * a5m + a4m, 0, 0, a6m, 0}};
  // ymij_p
  res[3] = {{0, a1m + a2c * a5m, 0,
             a1c * a5m * nix / niy + a3m * nix / niy + a6m * nix / niy,
             a1c * a5m + a2m + a3m + a6m,
             a1c * a5m * niz / niy + a3m * niz / niy + a6m * niz / niy,
             -a1c * a5m * nix / niy - a3m * nix / niy,
             -a1c * a5m * niz / niy - a3m * niz / niy, 0, a3c * a5m + a4m, 0,
             -a6m * nix / niy, -a6m * niz / niy}};
  // zmij_p
  res[4] = {{0, 0, a1m + a2c * a5m, 0, 0, a2m, 0, a1c * a5m + a3m, 0, 0,
             a3c * a5m + a4m, 0, a6m}};
  // xci_p
  res[5] = {{a2c, 0, 0, 0, 0, 0, a1c, 0, a3c, 0, 0, 0, 0}};
  // yci_p
  res[6] = {{0, a2c, 0, a1c * nix / niy, a1c, a1c * niz / niy, -a1c * nix / niy,
             -a1c * niz / niy, 0, a3c, 0, 0, 0}};
  // zci_p
  res[7] = {{0, 0, a2c, 0, 0, 0, 0, a1c, 0, 0, a3c, 0, 0}};

  return res;
}

std::array<std::array<double, 13>, 8>
pj_cone_z_max_deps(const std::array<double, 3> &ac,
                   const std::array<double, 6> &am, const Eigen::Vector3d &ni) {
  std::array<std::array<double, 13>, 8> res;

  const double nix = ni[0];
  const double niy = ni[1];
  const double niz = ni[2];

  const double a1c = ac[0];
  const double a2c = ac[1];
  const double a3c = ac[2];

  const double a1m = am[0];
  const double a2m = am[1];
  const double a3m = am[2];
  const double a4m = am[3];
  const double a5m = am[4];
  const double a6m = am[5];

  // zij
  res[0] = {{0, 0, 0, nix / niz, niy / niz, 1, -nix / niz, -niy / niz, 0, 0, 0,
             0, 0}};
  // zmij
  res[1] = {{0, 0, 0, nix / niz, niy / niz, 1, 0, 0, 0, 0, 0, -nix / niz,
             -niy / niz}};
  // xmij_p
  res[2] = {{a1m + a2c * a5m, 0, 0, a2m, 0, 0, a1c * a5m + a3m, 0,
             a3c * a5m + a4m, 0, 0, a6m, 0}};
  // ymij_p
  res[3] = {{0, a1m + a2c * a5m, 0, 0, a2m, 0, 0, a1c * a5m + a3m, 0,
             a3c * a5m + a4m, 0, 0, a6m}};
  // zmij_p
  res[4] = {{0, 0, a1m + a2c * a5m,
             a1c * a5m * nix / niz + a3m * nix / niz + a6m * nix / niz,
             a1c * a5m * niy / niz + a3m * niy / niz + a6m * niy / niz,
             a1c * a5m + a2m + a3m + a6m,
             -a1c * a5m * nix / niz - a3m * nix / niz,
             -a1c * a5m * niy / niz - a3m * niy / niz, 0, 0, a3c * a5m + a4m,
             -a6m * nix / niz, -a6m * niy / niz}};
  // xci_p
  res[5] = {{a2c, 0, 0, 0, 0, 0, a1c, 0, a3c, 0, 0, 0, 0}};
  // yci_p
  res[6] = {{0, a2c, 0, 0, 0, 0, 0, a1c, 0, a3c, 0, 0, 0}};
  // zci_p
  res[7] = {{0, 0, a2c, a1c * nix / niz, a1c * niy / niz, a1c, -a1c * nix / niz,
             -a1c * niy / niz, 0, 0, a3c, 0, 0}};

  return res;
}