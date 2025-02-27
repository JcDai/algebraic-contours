#include "apply_transformation.h"
#include "clough_tocher_surface.hpp"
#include "common.h"
#include "compute_boundaries.h"
#include "contour_network.h"
#include "generate_transformation.h"
#include "globals.cpp"
#include "twelve_split_spline.h"
#include <CLI/CLI.hpp>
#include <fstream>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <unsupported/Eigen/SparseExtra>

int main(int argc, char *argv[]) {
  // Build maps from strings to enums
  std::map<std::string, spdlog::level::level_enum> log_level_map{
      {"trace", spdlog::level::trace},       {"debug", spdlog::level::debug},
      {"info", spdlog::level::info},         {"warn", spdlog::level::warn},
      {"critical", spdlog::level::critical}, {"off", spdlog::level::off},
  };

  // Get command line arguments
  CLI::App app{"Generate Clough-Tocher cubic surface mesh."};
  std::string input_filename = "";
  std::string output_dir = "./";
  std::string output_name = "CT";
  std::string boundary_data = "";
  std::string vertex_normal_file = "";
  spdlog::level::level_enum log_level = spdlog::level::off;
  Eigen::Matrix<double, 3, 1> color = SKY_BLUE;
  int num_subdivisions = DISCRETIZATION_LEVEL;
  OptimizationParameters optimization_params;
  double weight = optimization_params.position_difference_factor;
  app.add_option("-i,--input", input_filename, "Mesh filepath")
      ->check(CLI::ExistingFile)
      ->required();
  app.add_option("--log_level", log_level, "Level of logging")
      ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case));
  app.add_option("--num_subdivisions", num_subdivisions,
                 "Number of subdivisions")
      ->check(CLI::PositiveNumber);
  app.add_option("-w,--weight", weight,
                 "Fitting weight for the quadratic surface approximation")
      ->check(CLI::PositiveNumber);
  app.add_option("-o, --output", output_name, "Output file prefix");
  app.add_option(
      "--boundary-data", boundary_data,
      "input boundary data. Only support 1 Function Value interpolant");
  app.add_option("--vertex_normals", vertex_normal_file,
                 "vertex normals in the order of lagrange nodes");
  CLI11_PARSE(app, argc, argv);

  // Set logger level
  spdlog::set_level(log_level);

  // Set optimization parameters
  optimization_params.position_difference_factor = weight;

  // Get input mesh
  Eigen::MatrixXd V, uv, N;
  Eigen::MatrixXi F, FT, FN;
  igl::readOBJ(input_filename, V, uv, N, F, FT, FN);

  // get boundary data
  bool have_external_boundary_data = false;
  std::vector<Eigen::Matrix<double, 12, 1>> ext_boundary_data;
  if (boundary_data != "") {
    have_external_boundary_data = true;
    std::ifstream boundary_data_file(boundary_data);
    for (int64_t i = 0; i < F.rows(); ++i) {
      Eigen::Matrix<double, 12, 1> bd;
      for (int j = 0; j < 12; ++j) {
        boundary_data_file >> bd(j, 0);
      }
      //   std::cout << bd << std::endl;
      ext_boundary_data.push_back(bd);
    }
    boundary_data_file.close();
  }

  // Generate quadratic spline
  spdlog::info("Computing spline surface");
  //   std::vector<std::vector<int>> face_to_patch_indices;
  //   std::vector<int> patch_to_face_indices;
  Eigen::SparseMatrix<double> fit_matrix;
  Eigen::SparseMatrix<double> energy_hessian;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
      energy_hessian_inverse;
  AffineManifold affine_manifold(F, uv, FT);

  CloughTocherSurface ct_surface(V, affine_manifold, optimization_params,
                                 fit_matrix, energy_hessian,
                                 energy_hessian_inverse);

  std::cout << "#F: " << ct_surface.m_affine_manifold.m_face_charts.size()
            << std::endl;
  std::cout << "#E: " << ct_surface.m_affine_manifold.m_edge_charts.size()
            << std::endl;
  std::cout << "#V: " << ct_surface.m_affine_manifold.m_vertex_charts.size()
            << std::endl;

  // important !!! call the following two before compute constraints, but only
  // once!!!
  ct_surface.m_affine_manifold.generate_lagrange_nodes();
  ct_surface.m_affine_manifold.compute_edge_global_uv_mappings();

  ct_surface.write_cubic_surface_to_msh_with_conn_from_lagrange_nodes(
      output_name + "_from_lagrange_nodes");

  ct_surface.write_connected_lagrange_nodes(output_name + "_bilaplacian_nodes",
                                            V);
  ct_surface.write_connected_lagrange_nodes_values(output_name +
                                                   "_bilaplacian_nodes_values");

  Eigen::SparseMatrix<double> c_f_int;
  ct_surface.C_F_int(c_f_int);
  Eigen::SparseMatrix<double> C_e_end, C_e_end_elim;
  ct_surface.C_E_end(C_e_end, C_e_end_elim);
  Eigen::SparseMatrix<double> C_e_mid;
  ct_surface.C_E_mid(C_e_mid);

  Eigen::MatrixXd v_normals;

  if (vertex_normal_file != "") {
    std::cout << "Reading external vertex normals file" << std::endl;
    std::vector<Eigen::Vector3d> lag_node_normals;
    std::ifstream lag_norm_file(vertex_normal_file);
    double x, y, z;
    while (lag_norm_file >> x >> y >> z) {
      lag_node_normals.emplace_back(x, y, z);
    }
    lag_norm_file.close();
    assert(lag_node_normals.size() ==
           ct_surface.m_affine_manifold.m_lagrange_nodes.size());
    if (lag_node_normals.size() !=
        ct_surface.m_affine_manifold.m_lagrange_nodes.size()) {
      std::cout
          << "Lagrange node size not compatible with lag normal size! expected "
          << ct_surface.m_affine_manifold.m_lagrange_nodes.size() << " but got "
          << lag_node_normals.size() << std::endl;
      throw std::runtime_error("normal size from file mismatching");
    }

    v_normals = Eigen::MatrixXd::Zero(V.rows(), 3);
    for (size_t i = 0; i < ct_surface.m_affine_manifold.m_lagrange_nodes.size();
         ++i) {
      if (ct_surface.m_affine_manifold.lagrange_node_to_v_map.find(i) !=
          ct_surface.m_affine_manifold.lagrange_node_to_v_map.end()) {
        v_normals.row(ct_surface.m_affine_manifold.lagrange_node_to_v_map[i]) =
            lag_node_normals[i];
      }
    }

  } else {
    // TODO: change weight
    igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA,
                            v_normals);
  }

  // // debug use
  // Eigen::MatrixXd test_normals(v_normals.rows(), v_normals.cols());
  // for (int64_t i = 0; i < v_normals.rows(); ++i) {
  //   for (int64_t j = 0; j < v_normals.cols(); ++j)
  //     if (j == 2) {
  //       test_normals(i, j) = 1;
  //     } else {
  //       test_normals(i, j) = 1;
  //     }
  // }

  // v_normals = test_normals;

  Eigen::SparseMatrix<double> c_cone;
  ct_surface.C_F_cone(c_cone, v_normals);

  // save sparse matrix
  Eigen::saveMarket(c_f_int, output_name + "_interior_constraint_matrix.txt");
  Eigen::saveMarket(C_e_end,
                    output_name + "_edge_endpoint_constraint_matrix.txt");
  Eigen::saveMarket(C_e_end_elim,
                    output_name +
                        "_edge_endpoint_constraint_matrix_eliminated.txt");
  Eigen::saveMarket(C_e_mid,
                    output_name + "_edge_midpoint_constraint_matrix.txt");
  Eigen::saveMarket(c_cone, output_name + "_cone_constraint_matrix.txt");

  // std::ofstream file(output_name + "_interior_constraint_matrix.txt");
  // file << std::setprecision(16) << c_f_int;
  // std::ofstream file_2(output_name + "_edge_endpoint_constraint_matrix.txt");
  // file_2 << std::setprecision(16) << C_e_end;
  // std::ofstream file_3(output_name + "_edge_midpoint_constraint_matrix.txt");
  // file_3 << std::setprecision(16) << C_e_mid;

  // std::ofstream file_4(output_name +
  //                      "_edge_endpoint_constraint_matrix_eliminated.txt");
  // file_4 << std::setprecision(16) << C_e_end_elim;

  // file.close();
  // file_2.close();
  // file_3.close();
  // file_4.close();

  if (have_external_boundary_data) {
    // TODO
    ct_surface
        .write_external_bd_interpolated_function_values_from_lagrange_nodes(
            output_name + "_function_values_from_lagrange_nodes",
            ext_boundary_data);
  }

  // check constraint error
  const auto &lag_values = ct_surface.m_lagrange_node_values;
  Eigen::MatrixXd lag_v_mat(lag_values.size(), 3);
  for (size_t i = 0; i < lag_values.size(); ++i) {
    lag_v_mat.row(i) = lag_values[i].transpose();
  }

  auto int_error = c_f_int * lag_v_mat;
  double int_max_error = int_error.maxCoeff();
  double int_min_error = int_error.minCoeff();
  std::cout << "interior max error: "
            << ((std::abs(int_max_error) > std::abs(int_min_error))
                    ? std::abs(int_max_error)
                    : std::abs(int_min_error))
            << std::endl;

  auto end_error = C_e_end * lag_v_mat;
  double end_max_error = end_error.maxCoeff();
  double end_min_error = end_error.minCoeff();
  std::cout << "endpoint max error: "
            << ((std::abs(end_max_error) > std::abs(end_min_error))
                    ? std::abs(end_max_error)
                    : std::abs(end_min_error))
            << std::endl;

  auto mid_error = C_e_mid * lag_v_mat;
  double mid_max_error = mid_error.maxCoeff();
  double mid_min_error = mid_error.minCoeff();
  std::cout << "midpoint max error: "
            << ((std::abs(mid_max_error) > std::abs(mid_min_error))
                    ? std::abs(mid_max_error)
                    : std::abs(mid_min_error))
            << std::endl;

  return 0;
}
