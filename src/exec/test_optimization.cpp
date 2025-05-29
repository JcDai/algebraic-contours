#include "optimize_clough_tocher.hpp"
#include <CLI/CLI.hpp>
#include <igl/readOBJ.h>

std::array<std::array<int64_t, 10>, 3>
get_micro_triangle_nodes(const AffineManifold &affine_manifold,
                         int64_t face_index) {
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

double evaluate_energy(double u[3], double v[3], double test_cp[3][10]) {
  // Build input mesh
  Eigen::MatrixXd V(3, 3);
  Eigen::MatrixXd uv(3, 2);
  Eigen::MatrixXi F(1, 3);
  Eigen::MatrixXi FT(1, 3);
  F << 0, 1, 2;
  FT << 0, 1, 2;
  for (int i = 0; i < 3; ++i) {
    uv.row(i) << u[i], v[i];
    V.row(i) << u[i], v[i], 0.;
  }
  AffineManifold affine_manifold(F, uv, FT);
  affine_manifold.generate_lagrange_nodes();

  // build initial surface
  spdlog::info("Computing spline surface");
  Eigen::SparseMatrix<double> fit_matrix;
  Eigen::SparseMatrix<double> energy_hessian;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
      energy_hessian_inverse;
  OptimizationParameters optimization_params;
  CloughTocherSurface ct_surface(V, affine_manifold, optimization_params,
                                 fit_matrix, energy_hessian,
                                 energy_hessian_inverse);
  ct_surface.write_cubic_surface_to_msh_with_conn_from_lagrange_nodes("temp",
                                                                      true);

  std::cout << "#F: " << ct_surface.m_affine_manifold.m_face_charts.size()
            << std::endl;
  std::cout << "#E: " << ct_surface.m_affine_manifold.m_edge_charts.size()
            << std::endl;
  std::cout << "#V: " << ct_surface.m_affine_manifold.m_vertex_charts.size()
            << std::endl;
  std::cout << "#nodes: " << ct_surface.m_lagrange_node_values.size()
            << std::endl;

  std::vector<Eigen::Vector3d> bezier_control_points =
      ct_surface.m_bezier_control_points;
  std::array<std::array<int64_t, 10>, 3> micro_nodes =
      get_micro_triangle_nodes(affine_manifold, 0);
  std::array<int64_t, 10> perm = {0, 9, 3, 4, 7, 8, 6, 2, 1, 5};
  for (int n = 0; n < 3; ++n) {
    for (int i = 0; i < 10; ++i) {
      int ni = micro_nodes[n][i];
      bezier_control_points[ni][0] = test_cp[n][perm[i]];
      bezier_control_points[ni][1] = 0;
      bezier_control_points[ni][2] = 0;
    }
  }
  spdlog::info("points are {}, {}, {}", bezier_control_points[0][0],
               bezier_control_points[1][0], bezier_control_points[2][0]);
  spdlog::info("center is {}", bezier_control_points[18][0]);

  CloughTocherOptimizer optimizer(V, F, affine_manifold);
  optimizer.fitting_weight = 0.;
  return optimizer.evaluate_energy(bezier_control_points);
}

int main(int argc, char *argv[]) {
  // Build maps from strings to enums
  std::map<std::string, spdlog::level::level_enum> log_level_map{
      {"trace", spdlog::level::trace},       {"debug", spdlog::level::debug},
      {"info", spdlog::level::info},         {"warn", spdlog::level::warn},
      {"critical", spdlog::level::critical}, {"off", spdlog::level::off},
  };

  // Get command line arguments
  CLI::App app{"Optimize Clough-Tocher cubic surface mesh."};
  std::string input_filename = "";
  std::string output_dir = "./";
  std::string output_name = "./CT";
  spdlog::level::level_enum log_level = spdlog::level::off;
  Eigen::Matrix<double, 3, 1> color = SKY_BLUE;
  app.add_option("-i,--input", input_filename, "Mesh filepath")
      ->check(CLI::ExistingFile)
      ->required();
  app.add_option("--log_level", log_level, "Level of logging")
      ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case));
  app.add_option("-o, --output", output_name, "Output file prefix");
  CLI11_PARSE(app, argc, argv);

  // Set logger level
  spdlog::set_level(log_level);

  // first test
  double u1[3] = {0.0, 1.0, 0.0};
  double v1[3] = {0.0, 0.0, 1.0};
  double test_cp1[3][10] = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  printf("E = %g, value should be 0\n", evaluate_energy(u1, v1, test_cp1));

  // second test
  double u2[3] = {-1.0, 1.0, 0.0};
  double v2[3] = {0.0, 0.0, 1.0};
  double test_cp2[3][10] = {
      {-3., -2., -1.296296296, 0., -1., -0.6666666667, 1.296296296,
       -0.3333333333, 3.333333333, 7.},
      {7., 3.333333333, 1.296296296, 0., 3.333333333, 1.222222222, 0., 1., 0.,
       0.},
      {0., 0., 0., 0., -1., -1.222222222, -1.296296296, -2., -2., -3.}};

  printf("E = %g, value from Maple 24\n", evaluate_energy(u2, v2, test_cp2));

  // third test
  double u3[3] = {-1, 2., 1 / 2.};
  double v3[3] = {1 / 2., -1 / 2., 3};
  double test_cp3[3][10] = {
      {-2.750000000, -1.208333333, -0.5833333333, 2.125000000, -0.3333333333,
       -0.4583333333, 5.500000000, 0.4166666667, 12.62500000, 23.50000000},
      {23.50000000, 12.62500000, 5.500000000, 2.125000000, 13.95833333,
       4.333333333, 1.458333333, 0.5000000000, 0.1250000000, -1.875000000},
      {-1.875000000, 0.1250000000, 1.458333333, 2.125000000, 1.750000000,
       -0.08333333333, -0.5833333333, -0.5416666667, -1.208333333,
       -2.750000000}};

  printf("E = %g, value from Maple 348\n", evaluate_energy(u3, v3, test_cp3));

  return 0;
}
