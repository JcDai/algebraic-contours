#pragma once
#include "common.h"

Eigen::Matrix<double, 12, 12> L_L2d_ind_m();
Eigen::Matrix<double, 7, 12> L_d2L_dep_m();
Eigen::Matrix<double, 5, 1> c_e_m();
std::array<std::array<Eigen::Matrix<double, 1, 12>, 3>, 3> c_t_m();

// bezier
Eigen::Matrix<double, 7, 1> c_hij_m();
Eigen::Matrix<double, 10, 10> p3_lag2bezier_m();
Eigen::Matrix<double, 10, 10> p3_bezier2lag_m();