#include "optimize_clough_tocher.hpp"
#include "polyscope/surface_mesh.h"
#include <CLI/CLI.hpp>
#include <igl/readOBJ.h>

/**
 * @brief Executable to take a triangle mesh file (stored as OBJ) and produce an
 * optimized C1 cubic spline.
 *
 */

// TODO: Obtained from affine_manifold.cpp. Make standalone function
const std::array<PlanarPoint, 19> CT_nodes = {{
    PlanarPoint(1., 0.),           // b0    0
    PlanarPoint(0., 1.),           // b1    1
    PlanarPoint(0., 0.),           // b2    2
    PlanarPoint(2. / 3., 1. / 3.), // b01   3
    PlanarPoint(1. / 3., 2. / 3.), // b10   4
    PlanarPoint(0., 2. / 3.),      // b12   5
    PlanarPoint(0., 1. / 3.),      // b21   6
    PlanarPoint(1. / 3., 0.),      // b20   7
    PlanarPoint(2. / 3., 0.),      // b02   8
    PlanarPoint(4. / 9., 4. / 9.), // b01^c 9
    PlanarPoint(1. / 9., 4. / 9.), // b12^c 10
    PlanarPoint(4. / 9., 1. / 9.), // b20^c 11
    PlanarPoint(7. / 9., 1. / 9.), // b0c   12
    PlanarPoint(5. / 9., 2. / 9.), // bc0   13
    PlanarPoint(1. / 9., 7. / 9.), // b1c   14
    PlanarPoint(2. / 9., 5. / 9.), // bc1   15
    PlanarPoint(1. / 9., 1. / 9.), // b2c   16
    PlanarPoint(2. / 9., 2. / 9.), // bc2   17
    PlanarPoint(1. / 3., 1. / 3.), // bc    18
}};

std::vector<Eigen::Vector3d>
generate_linear_clough_tocher_surface(CloughTocherSurface &ct_surface,
                                      const Eigen::MatrixXd &V) {
  int num_nodes = ct_surface.m_lagrange_node_values.size();
  std::vector<Eigen::Vector3d> lagrange_control_points(num_nodes);
  const auto &affine_manifold = ct_surface.m_affine_manifold;
  const auto &F = affine_manifold.get_faces();
  int num_faces = affine_manifold.num_faces();
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    const auto &l_nodes = face_chart.lagrange_nodes;
    int vi = F(fijk, 0);
    int vj = F(fijk, 1);
    int vk = F(fijk, 2);
    Eigen::Vector3d Vi = V.row(vi);
    Eigen::Vector3d Vj = V.row(vj);
    Eigen::Vector3d Vk = V.row(vk);
    for (int i = 0; i < 19; ++i) {
      double u = CT_nodes[i][0];
      double v = CT_nodes[i][1];
      double w = 1 - u - v;
      lagrange_control_points[l_nodes[i]] = u * Vi + v * Vj + w * Vk;
    }
  }

  Eigen::SparseMatrix<double, 1> l2b_mat;
  ct_surface.lag2bezier_full_mat(l2b_mat);

  Eigen::MatrixXd lagrange_matrix(num_nodes, 3);
  for (int64_t i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < 3; ++j) {
      lagrange_matrix(i, j) = lagrange_control_points[i][j];
    }
  }
  Eigen::MatrixXd bezier_matrix = l2b_mat * lagrange_matrix;
  std::vector<Eigen::Vector3d> bezier_control_points(num_nodes);
  for (int64_t i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < 3; ++j) {
      bezier_control_points[i][j] = bezier_matrix(i, j);
    }
  }

  return bezier_control_points;
}

// Helper function to write a curface with external bezier nodes to file
void write_mesh(CloughTocherSurface &ct_surface,
                const std::vector<Eigen::Vector3d> &bezier_control_points,
                const std::string &filename) {
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
  OptimizationParameters optimization_params;
  double weight = 1e3;
  double interpolation = 0.;
  int iterations = 1;
  int visualize = 0;
  app.add_option("-i,--input", input_filename, "Mesh filepath")
      ->check(CLI::ExistingFile)
      ->required();
  app.add_option("--log_level", log_level, "Level of logging")
      ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case));
  app.add_option("-w,--weight", weight,
                 "Fitting weight for the quadratic surface approximation")
      ->check(CLI::PositiveNumber);
  app.add_option("-n,--iterations", iterations,
                 "Number of iterations of optimization");
  app.add_option("-t,--interpolation", interpolation,
                 "Interpolation between original and final");
  app.add_option("-o, --output", output_name, "Output file prefix");
  app.add_option("-v, --visualize", visualize, "Visualize with polyscope");
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

  // build initial surface
  spdlog::info("Computing spline surface");
  Eigen::SparseMatrix<double> fit_matrix;
  Eigen::SparseMatrix<double> energy_hessian;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
      energy_hessian_inverse;
  CloughTocherSurface ct_surface(V, affine_manifold, optimization_params,
                                 fit_matrix, energy_hessian,
                                 energy_hessian_inverse);

  // WARNING: surface writing needed to generate points
  // TODO: make part of initialization
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

  // get initial bezier points
  std::vector<Eigen::Vector3d> bezier_control_points =
      ct_surface.m_bezier_control_points;
  write_mesh(ct_surface, bezier_control_points, "initial_mesh");

  // map bezier points to original positions
  // TODO: move this into internal optimizer function
  int num_faces = affine_manifold.num_faces();
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    const auto &nodes = face_chart.lagrange_nodes;
    bezier_control_points[nodes[0]] = V.row(F(fijk, 0));
    bezier_control_points[nodes[1]] = V.row(F(fijk, 1));
    bezier_control_points[nodes[2]] = V.row(F(fijk, 2));
  }

  bezier_control_points = generate_linear_clough_tocher_surface(ct_surface, V);
  write_mesh(ct_surface, bezier_control_points, "linear_mesh");

  // initialize optimizer
  CloughTocherOptimizer optimizer(V, F, affine_manifold);
  bezier_control_points =
      optimizer.optimize_laplacian_energy(bezier_control_points);
  optimizer.fitting_weight = weight;

  // optimize the bezier nodes with laplacian energy
  std::vector<Eigen::Vector3d> laplacian_control_points =
      optimizer.optimize_laplacian_energy(bezier_control_points);
  write_mesh(ct_surface, laplacian_control_points, "laplacian_mesh");
  std::vector<double> laplacian_energies = optimizer.compute_face_energies(
      bezier_control_points, optimizer.generate_laplacian_stiffness_matrix());

  // optimize the bezier nodes with laplace beltrami energy
  std::vector<Eigen::Vector3d> laplace_beltrami_control_points =
      optimizer.optimize_laplace_beltrami_energy(bezier_control_points,
                                                 iterations);
  write_mesh(ct_surface, laplace_beltrami_control_points,
             "laplace_beltrami_mesh");
  std::vector<double> lb_energies = optimizer.compute_face_energies(
      bezier_control_points,
      optimizer.generate_laplace_beltrami_stiffness_matrix());

  double laplacian_energy = 0.;
  double lb_energy = 0.;
  std::vector<double> energy_ratio(num_faces);
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    energy_ratio[fijk] = lb_energies[fijk] / laplacian_energies[fijk];
    laplacian_energy += laplacian_energies[fijk];
    lb_energy += lb_energies[fijk];
    // spdlog::info("face energies are {} and {}", laplacian_energies[fijk],
    // lb_energies[fijk]);
  }
  spdlog::info("total laplacian energy is {}", laplacian_energy);
  spdlog::info("total laplace beltrami energy is {}", lb_energy);

  // write lag2bezier mat for c1meshing soft constraint
  Eigen::SparseMatrix<double, 1> l2b_mat;
  ct_surface.lag2bezier_full_mat(l2b_mat);
  Eigen::saveMarket(l2b_mat, "CT_lag2bezier_matrix.txt");

  if (visualize != 0) {
    polyscope::init();
    polyscope::registerSurfaceMesh("mesh", V, F);
    polyscope::getSurfaceMesh("mesh")->addFaceScalarQuantity(
        "laplacian energy", laplacian_energies);
    polyscope::getSurfaceMesh("mesh")->addFaceScalarQuantity(
        "laplace beltrami energy", lb_energies);
    polyscope::getSurfaceMesh("mesh")->addFaceScalarQuantity("energy ratio",
                                                             energy_ratio);
    polyscope::show();
  }

  // optional interpolation (useful for debugging)
  std::vector<Eigen::Vector3d> interpolated_control_points(
      bezier_control_points.size());
  int node_cnt = bezier_control_points.size();
  for (int64_t i = 0; i < node_cnt; ++i) {
    for (int j = 0; j < 3; ++j) {
      interpolated_control_points[i][j] =
          t * bezier_control_points[i][j] +
          (1 - t) * laplace_beltrami_control_points[i][j];
    }
  }
  write_mesh(ct_surface, interpolated_control_points, "interpolated_mesh");

  return 0;
}
