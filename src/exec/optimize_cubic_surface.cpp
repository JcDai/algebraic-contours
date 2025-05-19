#include "optimize_clough_tocher.hpp"
#include <CLI/CLI.hpp>
#include <igl/readOBJ.h>

void write_mesh(
  CloughTocherSurface& ct_surface,
  const std::vector<Eigen::Vector3d>& bezier_control_points,
  const std::string& filename)
{
  Eigen::SparseMatrix<double, 1> b2l_mat;
  ct_surface.bezier2lag_full_mat(b2l_mat);

  int node_cnt = bezier_control_points.size();
  Eigen::MatrixXd bezier_matrix(node_cnt, 3);
  for (int64_t i = 0; i < node_cnt; ++i) {
    for (int j = 0; j < 3; ++j) {
      bezier_matrix(i, j) = bezier_control_points[i][j];
    }
  }

  Eigen::MatrixXd lagrange_matrix = b2l_mat * bezier_matrix;
  ct_surface.write_external_point_values_with_conn(filename, lagrange_matrix);
}

int main(int argc, char *argv[])
{
  // Build maps from strings to enums
  std::map<std::string, spdlog::level::level_enum> log_level_map{
      {"trace", spdlog::level::trace},
      {"debug", spdlog::level::debug},
      {"info", spdlog::level::info},
      {"warn", spdlog::level::warn},
      {"critical", spdlog::level::critical},
      {"off", spdlog::level::off},
  };

  // Get command line arguments
  CLI::App app{"Optimize Clough-Tocher cubic surface mesh."};
  std::string input_filename = "";
  std::string output_dir = "./";
  std::string output_name = "./CT";
  spdlog::level::level_enum log_level = spdlog::level::off;
  Eigen::Matrix<double, 3, 1> color = SKY_BLUE;
  OptimizationParameters optimization_params;
  double weight = 1e3;
  double interpolation = 0.;
  app.add_option("-i,--input", input_filename, "Mesh filepath")
      ->check(CLI::ExistingFile)
      ->required();
  app.add_option("--log_level", log_level, "Level of logging")
      ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case));
  app.add_option("-w,--weight", weight,
                 "Fitting weight for the quadratic surface approximation")
      ->check(CLI::PositiveNumber);
  app.add_option("-t,--interpolation", interpolation,
                 "Interpolation between original and final");
  app.add_option("-o, --output", output_name, "Output file prefix");
  CLI11_PARSE(app, argc, argv);
  double t = interpolation;

  // Set logger level
  spdlog::set_level(log_level);

  // Set optimization parameters
  optimization_params.position_difference_factor = weight;

  // Get input mesh
  Eigen::MatrixXd V, uv, N;
  Eigen::MatrixXi F, FT, FN;
  igl::readOBJ(input_filename, V, uv, N, F, FT, FN);
  AffineManifold affine_manifold(F, uv, FT);
  affine_manifold.generate_lagrange_nodes();
  affine_manifold.compute_edge_global_uv_mappings();
  //affine_manifold.view(V);

  // build initial surface
  spdlog::info("Computing spline surface");
  Eigen::SparseMatrix<double> fit_matrix;
  Eigen::SparseMatrix<double> energy_hessian;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
      energy_hessian_inverse;
  CloughTocherSurface ct_surface(V, affine_manifold, optimization_params,
                                 fit_matrix, energy_hessian,
                                 energy_hessian_inverse);
  ct_surface.write_cubic_surface_to_msh_with_conn_from_lagrange_nodes("temp", true);

  std::cout << "#F: " << ct_surface.m_affine_manifold.m_face_charts.size()
            << std::endl;
  std::cout << "#E: " << ct_surface.m_affine_manifold.m_edge_charts.size()
            << std::endl;
  std::cout << "#V: " << ct_surface.m_affine_manifold.m_vertex_charts.size()
            << std::endl;
  std::cout << "#nodes: " << ct_surface.m_lagrange_node_values.size()
            << std::endl;

	std::vector<Eigen::Vector3d> bezier_control_points = ct_surface.m_bezier_control_points;
  write_mesh(ct_surface, bezier_control_points, "initial_mesh");

  // map bezier points to original positions
  int num_faces = affine_manifold.num_faces();
  for (int fijk = 0; fijk < num_faces; ++fijk)
  {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    const auto &nodes = face_chart.lagrange_nodes;
    bezier_control_points[nodes[0]] = V.row(F(fijk, 0));
    bezier_control_points[nodes[1]] = V.row(F(fijk, 1));
    bezier_control_points[nodes[2]] = V.row(F(fijk, 2));
  }


  CloughTocherOptimizer optimizer(V, F, affine_manifold);
  optimizer.fitting_weight = weight;
  std::vector<Eigen::Vector3d> optimized_control_points = optimizer.optimize_energy(bezier_control_points);
  write_mesh(ct_surface, optimized_control_points, "optimized_mesh");

	std::vector<Eigen::Vector3d> interpolated_control_points(bezier_control_points.size());
  int node_cnt = bezier_control_points.size();
  for (int64_t i = 0; i < node_cnt; ++i) {
    for (int j = 0; j < 3; ++j) {
      interpolated_control_points[i][j] = t * bezier_control_points[i][j] + (1 - t) * optimized_control_points[i][j];
    }
  }

  return 0;
}
