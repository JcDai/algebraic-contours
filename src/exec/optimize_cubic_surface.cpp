#include "optimize_clough_tocher.hpp"
#include "polyscope/surface_mesh.h"
#include <CLI/CLI.hpp>
#include <igl/readOBJ.h>
#include <igl/upsample.h>
#include <igl/writeOBJ.h>

/**
 * @brief Executable to take a triangle mesh file (stored as OBJ) and produce an
 * optimized C1 cubic spline.
 *
 */

void
refine_mesh(Eigen::MatrixXd& V,
            Eigen::MatrixXi& F,
            Eigen::MatrixXd& uv,
            Eigen::MatrixXi& FT,
            int refinement)
{
  igl::upsample(V, F, refinement);
  igl::upsample(uv, FT, refinement);
}

int
main(int argc, char* argv[])
{
  // Build maps from strings to enums
  std::map<std::string, spdlog::level::level_enum> log_level_map{
    { "trace", spdlog::level::trace },       { "debug", spdlog::level::debug },
    { "info", spdlog::level::info },         { "warn", spdlog::level::warn },
    { "critical", spdlog::level::critical }, { "off", spdlog::level::off },
  };

  // Get command line arguments
  CLI::App app{ "Optimize Clough-Tocher cubic surface mesh." };
  std::string input_filename = "";
  std::string output_name = "./";
  std::string render_path = "./render.png";
  spdlog::level::level_enum log_level = spdlog::level::warn;
  Eigen::Matrix<double, 3, 1> color = SKY_BLUE;
  OptimizationParameters optimization_params;
  double weight = 1e3;
  int iterations = 1;
  double scale = 1.;
  bool visualize = false;
  int refinement = 0;
  bool invert_area = false;
  bool square_area = false;
  bool normalize_count = false;
  app.add_option("-i,--input", input_filename, "Mesh filepath")
    ->check(CLI::ExistingFile)
    ->required();
  app.add_option("--render_path", render_path, "Render output filepath");
  app.add_option("--log_level", log_level, "Level of logging")
    ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case));
  app
    .add_option("-w,--weight",
                weight,
                "Fitting weight for the quadratic surface approximation")
    ->check(CLI::PositiveNumber);
  app.add_option(
    "-n,--iterations", iterations, "Number of iterations of optimization");
  app.add_option("--scale", scale, "Scale input mesh");
  app.add_option("--refinement", refinement, "Levels of refinement");
  app.add_option("-o, --output", output_name, "Output file prefix");
  app.add_flag("-v, --visualize", visualize, "Visualize with polyscope");
  app.add_flag(
    "--invert_area", invert_area, "Use inverse area for fitting noramlization");
  app.add_flag("--square_area", square_area, "Use squared area in laplacian");
  app.add_flag("--normalize_count", normalize_count, "Normalize");
  CLI11_PARSE(app, argc, argv);
  std::string mesh_name =
    std::filesystem::path(input_filename).filename().replace_extension();

  // Set logger level
  spdlog::set_level(log_level);
  std::filesystem::create_directory(output_name);

  // Set optimization parameters
  optimization_params.position_difference_factor = weight;

  // Get input mesh
  Eigen::MatrixXd V, uv, N;
  Eigen::MatrixXi F, FT, FN;
  igl::readOBJ(input_filename, V, uv, N, F, FT, FN);
  spdlog::info("{} vertices", V.rows());
  if (scale != 1.)
    V *= scale;

  refine_mesh(V, F, uv, FT, refinement);

  Eigen::VectorXd double_area;
  igl::doublearea(V, F, double_area);
  double area = double_area.sum() / 2.;
  igl::doublearea(uv, FT, double_area);
  double uv_area = double_area.sum() / 2.;
  spdlog::info("embedding area: {}", area);
  spdlog::info("uv area: {}", uv_area);

  // normalize area
  spdlog::info("normalizing uv area");
  uv *= std::sqrt(area / uv_area);
  igl::doublearea(uv, FT, double_area);
  uv_area = double_area.sum() / 2.;
  spdlog::info("new uv area: {}", uv_area);

  AffineManifold affine_manifold(F, uv, FT);
  affine_manifold.generate_lagrange_nodes();

  // build initial surface
  spdlog::info("Computing spline surface");
  Eigen::SparseMatrix<double> fit_matrix;
  Eigen::SparseMatrix<double> energy_hessian;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
    energy_hessian_inverse;
  CloughTocherSurface ct_surface(V,
                                 affine_manifold,
                                 optimization_params,
                                 fit_matrix,
                                 energy_hessian,
                                 energy_hessian_inverse);
  // WARNING: surface writing needed to generate points
  // TODO: make part of initialization
  ct_surface.write_cubic_surface_to_msh_with_conn_from_lagrange_nodes(join_path(output_name, "initial"),
                                                                      true);
  ct_surface.write_degenerate_cubic_surface_to_msh_with_conn(
    join_path(output_name, "CT_degenerate_cubic_bezier_points"), V, F);
  std::vector<Eigen::Vector3d> bezier_control_points =
    generate_linear_clough_tocher_surface(ct_surface, V);
  write_mesh(
    ct_surface, bezier_control_points, join_path(output_name, "linear"));

  // initialize optimizer
  CloughTocherOptimizer optimizer(V, F, affine_manifold);
  optimizer.fitting_weight = weight;
  optimizer.invert_area = invert_area;
  optimizer.double_area = square_area;
  optimizer.normalize_count = normalize_count;
  optimizer.output_dir = output_name;

  // optimize the bezier nodes with laplacian energy
  std::vector<Eigen::Vector3d> laplacian_control_points =
    optimizer.optimize_laplacian_energy(bezier_control_points);
  write_mesh(ct_surface,
             laplacian_control_points,
             join_path(output_name, "laplacian_mesh"));

  // optimize the bezier nodes with laplace beltrami energy
  std::vector<Eigen::Vector3d> laplace_beltrami_control_points =
    optimizer.optimize_laplace_beltrami_energy(bezier_control_points,
                                               iterations);
  write_mesh(ct_surface,
             laplace_beltrami_control_points,
             join_path(output_name, "laplace_beltrami_mesh"));

  set_bezier_control_points(ct_surface, laplace_beltrami_control_points);
  Eigen::MatrixXd V_out;
  Eigen::MatrixXi F_out;
  ct_surface.discretize(3, V_out, F_out);
  igl::writeOBJ(join_path(output_name, "triangulated_mesh.obj"), V_out, F_out);

  std::vector<SpatialVector> points;
  std::vector<std::vector<int>> polylines;
  ct_surface.discretize_patch_boundaries(3, points, polylines, true);
  write_polylines_to_obj(
    join_path(output_name, "patch_boundaries.obj"), points, polylines);
  ct_surface.discretize_patch_boundaries(3, points, polylines, false);
  write_polylines_to_obj(
    join_path(output_name, "interior_patch_boundaries.obj"), points, polylines);

  // write lag2bezier mat for c1meshing soft constraint
  Eigen::SparseMatrix<double, 1> l2b_mat;
  ct_surface.lag2bezier_full_mat(l2b_mat);
  Eigen::saveMarket(l2b_mat, join_path(output_name, "CT_lag2bezier_matrix.txt"));

  ct_surface.add_surface_to_viewer({ 1, 0.4, 0.3 }, 3);
  polyscope::registerSurfaceMesh("PL mesh", V, F);
  polyscope::screenshot(render_path);
  polyscope::screenshot(join_path(output_name, "render.png"));
  if (visualize) {
    polyscope::show();
  }

  return 0;
}
