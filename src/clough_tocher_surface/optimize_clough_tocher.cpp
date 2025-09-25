#include "optimize_clough_tocher.hpp"
#include "autogen/Clough_Tocher_Laplace_Beltrami.c"
#include "autogen/Clough_Tocher_Laplace_Beltrami_autodiff.c"
#include "autogen/Clough_Tocher_Laplacian.c"
#include "clough_tocher_constraint_matrices.hpp"
#include "igl/doublearea.h"

CloughTocherOptimizer::CloughTocherOptimizer(
  const Eigen::MatrixXd V,
  const Eigen::MatrixXi F,
  const AffineManifold affine_manifold,
  bool use_incenter)
  : fitting_weight(1e5)
  , m_V(V)
  , m_F(F)
  , m_affine_manifold(affine_manifold)
  , m_use_incenter(use_incenter)
{
  // TODO: Would be better to avoid the uneccesary construction of a surface
  Eigen::SparseMatrix<double> fit_matrix;
  Eigen::SparseMatrix<double> energy_hessian;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
    energy_hessian_inverse;
  OptimizationParameters optimization_params;
  ct_surface = CloughTocherSurface(V,
                                   affine_manifold,
                                   optimization_params,
                                   fit_matrix,
                                   energy_hessian,
                                   energy_hessian_inverse);

  // build constraint and projection matrices
  timer.start();
  initialize_ind_to_full_matrices(m_use_incenter);
  spdlog::info("constraint matrix construction took {} s",
               timer.getElapsedTime());

  // build energy matrices
  timer.start();
  m_stiffness_matrix = generate_laplacian_stiffness_matrix();
  m_position_matrix = generate_position_matrix();
  spdlog::info("energy matrix construction took {} s", timer.getElapsedTime());
}

Eigen::VectorXd
CloughTocherOptimizer::project_to_reduced_subspace(
  const Eigen::VectorXd& p0) const
{
  if (use_orthogonal_projection) {
    // project to reduced space and up to satisfy constraints
    const Eigen::SparseMatrix<double>& C = get_ind_to_full_matrix();
    const Eigen::SparseMatrix<double>& M = C.transpose() * C;
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(M);
    return solver.solve(C.transpose() * p0);
  } else {
    const Eigen::SparseMatrix<double>& F = get_full_to_ind_matrix();
    return F * p0;
  }
}

std::vector<Eigen::Vector3d>
CloughTocherOptimizer::project_to_constraints(
  const std::vector<Eigen::Vector3d>& bezier_control_points) const
{
  // build initial position vector
  Eigen::VectorXd p0 = build_node_vector(bezier_control_points);

  // project to reduced space and back up to full space to satisfy constraints
  const Eigen::SparseMatrix<double>& C = get_ind_to_full_matrix();
  Eigen::VectorXd p = C * project_to_reduced_subspace(p0);

  return build_control_points(p);
}

void
CloughTocherOptimizer::initialize_data_log()
{
  // Generate data log path
  std::filesystem::create_directory(output_dir);
  std::string data_log_path;

  // Open main logging file
  data_log_path = join_path(output_dir, "iteration_log.csv");
  spdlog::info("Writing log to {}", data_log_path);

  log_file = std::ofstream(data_log_path, std::ios::out | std::ios::trunc);
  log_file << "num_iter,";
  log_file << "initial_energy,";
  log_file << "energy_decrease,";
  log_file << "step_size,";
  log_file << "total_time,";
  log_file << "assemble_time,";
  log_file << "solve_time,";
  log_file << "solve_residual,";
  log_file << "constraint_error,";
  log_file << std::endl;
}

// Write newton log iteration data to file
void
CloughTocherOptimizer::write_data_log_entry()
{
  log_file << ID.iter << ",";
  log_file << std::fixed << std::setprecision(17) << ID.initial_energy << ",";
  log_file << std::fixed << std::setprecision(17)
           << ID.initial_energy - ID.optimized_energy << ",";
  log_file << std::scientific << std::setprecision(6) << ID.step_size << ",";
  log_file << std::fixed << std::setprecision(6) << ID.total_time << ",";
  log_file << std::fixed << std::setprecision(6) << ID.assemble_time << ",";
  log_file << std::fixed << std::setprecision(6) << ID.solve_time << ",";
  log_file << std::scientific << std::setprecision(6) << ID.solve_residual
           << ",";
  log_file << std::scientific << std::setprecision(6) << ID.constraint_error
           << ",";
  log_file << std::endl;
}

// close the log file
void
CloughTocherOptimizer::close_logs()
{
  log_file.close();
}

std::vector<Eigen::Vector3d>
CloughTocherOptimizer::optimize_laplacian_energy(
  const std::vector<Eigen::Vector3d>& bezier_control_points)
{
  // build initial position vector
  Eigen::VectorXd p0 = build_node_vector(bezier_control_points);

  // compute hessian
  timer.start();
  double k = compute_normalized_fitting_weight();
  const Eigen::SparseMatrix<double>& C = get_ind_to_full_matrix();
  const Eigen::SparseMatrix<double>& F = get_full_to_ind_matrix();
  const Eigen::SparseMatrix<double>& hessian_smooth = get_stiffness_matrix();
  const Eigen::SparseMatrix<double>& P = get_position_matrix();
  Eigen::SparseMatrix<double> hessian =
    C.transpose() * ((hessian_smooth + k * P) * C);
  spdlog::info("matrix construction took {} s", timer.getElapsedTime());

  // invert hessian
  timer.start();
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> hessian_inverse;
  hessian_inverse.compute(hessian);
  spdlog::info("matrix solve took {} s", timer.getElapsedTime());

  // get base energy
  double E0 = 0.5 * k * p0.dot(p0);

  // get derivative
  Eigen::VectorXd derivative = -k * C.transpose() * (P * p0);

  // print initial energy
  // Eigen::VectorXd N0 = F * p0;
  Eigen::VectorXd N0 = project_to_reduced_subspace(p0);
  spdlog::info("initial fit energy: {}",
               evaluate_quadratic_energy(
                 C.transpose() * ((k * P) * C), derivative, E0, N0));
  spdlog::info("initial smoothness energy: {}",
               evaluate_quadratic_energy(
                 C.transpose() * (hessian_smooth * C), 0 * derivative, 0., N0));
  spdlog::info("initial energy is {}",
               evaluate_quadratic_energy(hessian, derivative, E0, N0));

  // solve for optimal solution
  Eigen::VectorXd N = -hessian_inverse.solve(derivative);
  Eigen::VectorXd p = C * N;
  Eigen::VectorXd res = (hessian * N) + derivative;
  spdlog::info("optimized energy is {}",
               evaluate_quadratic_energy(hessian, derivative, E0, N));
  spdlog::info("residual error is {}", res.cwiseAbs().maxCoeff());

  // check that solution satisfies constraints
  Eigen::VectorXd pr = C * (F * p);
  spdlog::info("constraint reconstruction error is {}",
               (pr - p).cwiseAbs().maxCoeff());

  return build_control_points(p);
}

std::vector<double>
CloughTocherOptimizer::compute_face_energies(
  const std::vector<Eigen::Vector3d>& bezier_control_points,
  bool use_laplace_beltrami)
{
  const std::vector<Eigen::Vector3d>& p = bezier_control_points;

  // assemble IJV matrix entries
  const auto& affine_manifold = get_affine_manifold();
  int num_faces = affine_manifold.num_faces();
  std::vector<double> face_energies(num_faces);
  Eigen::SparseMatrix<double> stiffness_matrix;
  stiffness_matrix.resize(19, 19);
  std::vector<Eigen::Vector3d> local_control_points(19);
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    std::vector<Triplet> stiffness_matrix_trips;
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    const auto& l_nodes = face_chart.lagrange_nodes;
    std::array<std::array<int64_t, 10>, 3> local_nodes =
      get_local_micro_triangle_nodes();
    if (use_laplace_beltrami) {
      for (int i = 0; i < 19; ++i) {
        local_control_points[i] = bezier_control_points[l_nodes[i]];
      }

      assemble_local_laplace_beltrami_siffness_matrix(
        local_control_points, local_nodes, stiffness_matrix_trips);
    } else {
      assemble_local_laplacian_siffness_matrix(
        face_chart.face_uv_positions, local_nodes, stiffness_matrix_trips);
    }

    // build matrix
    stiffness_matrix.setFromTriplets(stiffness_matrix_trips.begin(),
                                     stiffness_matrix_trips.end());

    for (int i = 0; i < 19; ++i) {
      for (int j = 0; j < 19; ++j) {
        for (int d = 0; d < 3; ++d) {
          int I = l_nodes[i];
          int J = l_nodes[j];
          face_energies[fijk] +=
            0.5 * p[I][d] * p[J][d] * stiffness_matrix.coeff(i, j);
        }
      }
    }
  }

  return face_energies;
}

double
CloughTocherOptimizer::compute_normalized_fitting_weight() const
{
  const auto& V = get_vertices();
  const auto& faces = get_faces();

  // begin with just the base fitting weight
  double normalized_fitting_weight = fitting_weight;

  // normalize by area (inverted or not)
  Eigen::VectorXd double_area;
  igl::doublearea(V, faces, double_area);
  double area = double_area.sum() / 2.;
  if (invert_area) {
    normalized_fitting_weight /= area;
  } else {
    normalized_fitting_weight *= area;
  }

  // optionally normalize by the vertex count
  if (normalize_count) {
    int num_vertices = V.rows();
    normalized_fitting_weight /= num_vertices;
  }

  return normalized_fitting_weight;
}

void
CloughTocherOptimizer::checkpoint_control_points(
  const std::vector<Eigen::Vector3d>& bezier_control_points,
  int iter)
{
  polyscope::removeAllStructures();
  std::filesystem::create_directory(output_dir);
  set_bezier_control_points(ct_surface, bezier_control_points);
  write_mesh(ct_surface,
             bezier_control_points,
             join_path(output_dir, "iter_" + std::to_string(iter)));
  ct_surface.add_surface_to_viewer({ 0.1, 0.1, 0.8 }, 3, "laplace_beltrami");
  polyscope::screenshot(
    join_path(output_dir, "iter_" + std::to_string(iter) + ".png"));
}

std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
CloughTocherOptimizer::generate_position_energy_quadratic(
  const std::vector<Eigen::Vector3d>& bezier_control_points,
  const std::vector<Eigen::Vector3d>& optimized_control_points) const
{
  Eigen::VectorXd p_init = build_node_vector(bezier_control_points);
  Eigen::VectorXd p = build_node_vector(optimized_control_points);
  Eigen::VectorXd d = p - p_init;

  Eigen::SparseMatrix<double> P = generate_position_matrix(d);
  Eigen::VectorXd g = P * d;
  double energy = g.dot(d);
  P *= p_norm * (p_norm - 1);
  g *= p_norm;
  return std::make_tuple(energy, g, P);
}

std::vector<Eigen::Vector3d>
CloughTocherOptimizer::optimize_laplace_beltrami_energy(
  const std::vector<Eigen::Vector3d>& bezier_control_points,
  int iterations,
  double step_size)
{
  total_timer.start();
  initialize_data_log();

  // build initial position vector
  Eigen::VectorXd p_init = build_node_vector(bezier_control_points);

  // get fixed matrices
  double k = compute_normalized_fitting_weight();
  spdlog::info("Using normalized fitting weight {}", k);
  const Eigen::SparseMatrix<double>& C = get_ind_to_full_matrix();
  const Eigen::SparseMatrix<double>& F = get_full_to_ind_matrix();
  // Eigen::SparseMatrix<double> P = generate_position_matrix();

  Eigen::VectorXd N0 = project_to_reduced_subspace(p_init);
  std::vector<Eigen::Vector3d> optimized_control_points =
    build_control_points(C * N0);
  Eigen::VectorXd p0 = C * N0;

  // get base energy
  // P = generate_position_matrix(p0 - p_init);
  // Eigen::VectorXd d_fit = -k * (P * p_init);
  double energy_fit, energy_smooth;
  Eigen::VectorXd derivative_fit, derivative_smooth, derivative;
  Eigen::SparseMatrix<double> hessian_fit, hessian_smooth, hessian;
  // energy_fit = 0.5 * k * p_init.dot(p_init);
  // derivative_fit = k * (P * (p0 - p_init));
  // derivative = C.transpose() * (derivative_fit + derivative_smooth);

  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> hessian_inverse;
  // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> hessian_inverse;

  timer.start();
  std::tie(energy_fit, derivative_fit, hessian_fit) =
    generate_position_energy_quadratic(bezier_control_points,
                                       optimized_control_points);
  if (use_parametric_metric) {
    // TODO: remove factor of 2 once quadrature fixed
    hessian_smooth = generate_laplace_beltrami_stiffness_matrix() / 2.;
    // hessian_smooth = generate_laplacian_stiffness_matrix();
  } else if (false) {
    std::tie(energy_smooth, derivative_smooth, hessian_smooth) =
      generate_autodiff_laplace_beltrami_stiffness_matrix(
        optimized_control_points);
  } else if (use_fixed_metric) {
    hessian_smooth =
      generate_laplace_beltrami_stiffness_matrix(optimized_control_points);
    derivative_smooth = hessian_smooth * p0;
    energy_smooth = 0.5 * p0.dot(derivative_smooth);
  } else {
    std::tie(energy_smooth, derivative_smooth) =
      generate_autodiff_laplace_beltrami_gradient(optimized_control_points);
    hessian_smooth =
      generate_laplace_beltrami_stiffness_matrix(optimized_control_points);
  }
  derivative = C.transpose() * (derivative_smooth + k * derivative_fit);
  hessian = C.transpose() * ((hessian_smooth + k * hessian_fit) * C);
  hessian_inverse.compute(hessian);

  Eigen::VectorXd N1 = -hessian_inverse.solve(derivative);
  Eigen::VectorXd res = (hessian * N1) + derivative;
  ID.solve_residual = res.cwiseAbs().maxCoeff();

  spdlog::info("initial fit energy: {}", energy_fit);
  spdlog::info("initial smoothness energy: {}", energy_smooth);

  // ID.initial_energy = std::numeric_limits<double>::infinity();
  // ID.initial_energy = evaluate_quadratic_energy(hessian, derivative, E0, N0);
  ID.initial_energy = energy_smooth + k * energy_fit;
  spdlog::info("initial energy: {}", ID.initial_energy);
  double max_res_error = 10. * ID.solve_residual;
  ID.step_size = step_size;
  for (ID.iter = 1; ID.iter < iterations + 1; ++ID.iter) {
    // solve for optimal solution
    Eigen::VectorXd g = -derivative;
    Eigen::VectorXd N1 = -hessian_inverse.solve(derivative);
    Eigen::VectorXd p1 = C * N1;
    Eigen::VectorXd res = (hessian * N1) + derivative;
    ID.solve_residual = res.cwiseAbs().maxCoeff();
    // Eigen::VectorXd d = N1 - N0;
    Eigen::VectorXd d = N1;
    spdlog::info("newton decr: {}", d.dot(derivative));

    spdlog::info(
      "iter {}: E={}, res={}", ID.iter, ID.initial_energy, ID.solve_residual);

    // do line search
    // ID.step_size = (ID.iter == 1) ? 1.0 : 0.5;
    ID.step_size = std::min(2 * ID.step_size, step_size);
    ID.optimized_energy = ID.initial_energy;
    Eigen::VectorXd N, p;
    while (true) {
      // interpolate in reduced space and project to full constrol points
      N = N0 + (ID.step_size * d);
      p = C * N;
      optimized_control_points = build_control_points(p);
      // spdlog::debug("energy in previous metric: {}",
      // evaluate_quadratic_energy(hessian, derivative, E0, N));

      // compute hessian
      timer.start();
      std::tie(energy_fit, derivative_fit, hessian_fit) =
        generate_position_energy_quadratic(bezier_control_points,
                                           optimized_control_points);
      // derivative_fit = k * (P * (p - p_init));
      if (false) {
        // hessian_smooth =
        // generate_laplace_beltrami_stiffness_matrix(optimized_control_points);
        std::tie(energy_smooth, derivative_smooth, hessian_smooth) =
          generate_autodiff_laplace_beltrami_stiffness_matrix(
            optimized_control_points);
      } else if (use_fixed_metric) {
        hessian_smooth =
          generate_laplace_beltrami_stiffness_matrix(optimized_control_points);
        derivative_smooth = hessian_smooth * p;
        energy_smooth = 0.5 * p.dot(derivative_smooth);
      } else {
        std::tie(energy_smooth, derivative_smooth) =
          generate_autodiff_laplace_beltrami_gradient(optimized_control_points);
        hessian_smooth =
          generate_laplace_beltrami_stiffness_matrix(optimized_control_points);
      }
      derivative = C.transpose() * (derivative_smooth + k * derivative_fit);
      hessian = C.transpose() * ((hessian_smooth + k * hessian_fit) * C);
      ID.assemble_time = timer.getElapsedTime();

      // compute optimized energy
      ID.optimized_energy = energy_smooth + k * energy_fit;

      // invert hessian
      timer.start();
      hessian_inverse.compute(hessian);
      ID.solve_time = timer.getElapsedTime();

      // compute residual error in next step
      Eigen::VectorXd N_next = -hessian_inverse.solve(derivative);
      Eigen::VectorXd res = (hessian * N_next) + derivative;
      ID.solve_residual = res.cwiseAbs().maxCoeff();

      // write log
      spdlog::info("step {}: delta E={}, res={}",
                   ID.step_size,
                   ID.initial_energy - ID.optimized_energy,
                   ID.solve_residual);

      // check convergence criteria
      if ((!bound_energy || (ID.optimized_energy <= ID.initial_energy)) &&
          (!bound_residual || (ID.solve_residual <= max_res_error)))
        break;
      if (ID.step_size < 1e-50) {
        spdlog::info("switching to gradient");
        d = g;
      }
      if (ID.step_size < 1e-10)
        break;

      // reduce step size and continue
      ID.step_size = ID.step_size / 2.;
    }

    // check that solution satisfies constraints
    Eigen::VectorXd pr = C * (F * p);
    ID.constraint_error = (pr - p).cwiseAbs().maxCoeff();
    if (ID.constraint_error > 1e-10) {
      spdlog::warn("constraint reconstruction error is {}",
                   ID.constraint_error);
    }

    // end iteration log output
    spdlog::info("matrix assembly took {} s, solve took {} s\n",
                 ID.assemble_time,
                 ID.solve_time);

    ID.total_time = total_timer.getElapsedTime();
    write_data_log_entry();

    N0 = N;
    p0 = p;
    ID.initial_energy = ID.optimized_energy;
    max_res_error =
      std::max(1e-4, ID.solve_residual * 10); // allow order of magnitude growth

    // exit if done
    if (ID.step_size < 1e-10)
      break;

    // serialize if checkpoint iteration
    int checkpoint = 10;
    if (((ID.iter % checkpoint) == 0) || (ID.iter < 0)) {
      checkpoint_control_points(optimized_control_points, ID.iter);
    }
  }

  spdlog::info("final energy: {}", ID.optimized_energy);
  close_logs();

  return optimized_control_points;
}

std::vector<Eigen::Vector3d>
CloughTocherOptimizer::gradient_descent_laplace_beltrami_energy(
  const std::vector<Eigen::Vector3d>& bezier_control_points,
  int iterations,
  double step_size)
{
  total_timer.start();
  initialize_data_log();

  // build initial position vector
  Eigen::VectorXd p_init = build_node_vector(bezier_control_points);

  // get fixed matrices
  double k = compute_normalized_fitting_weight();
  spdlog::info("Using normalized fitting weight {}", k);
  const Eigen::SparseMatrix<double>& C = get_ind_to_full_matrix();
  const Eigen::SparseMatrix<double>& P = get_position_matrix();

  // get base energy
  double E0 = 0.5 * k * p_init.dot(p_init);

  Eigen::VectorXd N0 = project_to_reduced_subspace(p_init);
  std::vector<Eigen::Vector3d> optimized_control_points =
    build_control_points(C * N0);
  Eigen::VectorXd p0 = C * N0;

  // get derivative
  double energy_smooth = 0.;
  Eigen::VectorXd d_fit = -k * (P * p_init);
  Eigen::VectorXd derivative_fit = k * (P * (p0 - p_init));
  Eigen::VectorXd derivative_smooth = Eigen::VectorXd::Zero(p_init.size());
  Eigen::VectorXd derivative =
    C.transpose() * (derivative_fit + derivative_smooth);

  for (ID.iter = 1; ID.iter < iterations + 1; ++ID.iter) {
    // solve for optimal solution
    derivative_fit = k * (P * (p0 - p_init));
    std::tie(energy_smooth, derivative_smooth) =
      generate_autodiff_laplace_beltrami_gradient(optimized_control_points);
    derivative = C.transpose() * (derivative_smooth + derivative_fit);

    Eigen::VectorXd d = -derivative;
    spdlog::info("iter {}: E={}", ID.iter, ID.initial_energy);
    ID.step_size = step_size;
    Eigen::VectorXd N, p;
    N = N0 + (ID.step_size * d);
    p = C * N;
    optimized_control_points = build_control_points(p);

    // compute optimized energy
    ID.optimized_energy =
      energy_smooth + evaluate_quadratic_energy(k * P, d_fit, E0, C * N);

    ID.total_time = total_timer.getElapsedTime();
    write_data_log_entry();

    N0 = N;
    p0 = p;
    ID.initial_energy = ID.optimized_energy;
  }

  spdlog::info("final energy: {}", ID.optimized_energy);
  close_logs();

  return optimized_control_points;
}

double
CloughTocherOptimizer::evaluate_energy(
  const std::vector<Eigen::Vector3d>& bezier_control_points)
{
  // build initial position vector
  Eigen::VectorXd p0 = build_node_vector(bezier_control_points);

  // compute hessian
  double k = compute_normalized_fitting_weight();
  const Eigen::SparseMatrix<double>& A = get_stiffness_matrix();
  const Eigen::SparseMatrix<double>& P = get_position_matrix();
  Eigen::SparseMatrix<double> hessian = (A + k * P);

  // get base energy
  double E0 = 0.5 * k * p0.dot(p0);

  // get derivative
  Eigen::VectorXd derivative = -k * (P * p0);

  // compute energy
  double E = evaluate_quadratic_energy(hessian, derivative, E0, p0);

  return E;
}

void
assign_spvec_to_spmat_row_help(Eigen::SparseMatrix<double, 1>& mat,
                               const Eigen::SparseVector<double>& vec,
                               const int row)
{
  for (Eigen::SparseVector<double>::InnerIterator it(vec); it; ++it) {
    mat.coeffRef(row, it.index()) = it.value();
  }
}

void
CloughTocherOptimizer::initialize_ind_to_full_matrices(bool use_incenter)
{
  const auto& V = get_vertices();
  const auto& F = get_faces();

  // TODO: Add option to pass in
  Eigen::MatrixXd v_normals;
  igl::per_vertex_normals(
    V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, v_normals);

  // build cone constraint system
  int64_t node_cnt = ct_surface.m_affine_manifold.m_lagrange_nodes.size();
  Eigen::SparseMatrix<double, Eigen::RowMajor> f2f_expanded(node_cnt * 3,
                                                            node_cnt * 3);
  f2f_expanded.reserve(Eigen::VectorXi::Constant(node_cnt * 3, 40));
  std::vector<int> independent_node_map(node_cnt * 3, -1);
  std::vector<bool> node_assigned(node_cnt, false);

  if (!use_incenter) {
    std::cout << "compute cone constraints ..." << std::endl;
    ct_surface.bezier_cone_constraints_expanded(
      f2f_expanded, independent_node_map, node_assigned, v_normals);
  }

  bool debug_isolate = false; // TODO: set to true only for debugging

  std::cout << "compute endpoint constraints ..." << std::endl;
  ct_surface.bezier_endpoint_ind2dep_expanded(
    f2f_expanded, independent_node_map, debug_isolate);

  std::cout << "compute interior 1 constraints ..." << std::endl;
  ct_surface.bezier_internal_ind2dep_1_expanded(
    f2f_expanded, independent_node_map, use_incenter);

  std::cout << "compute midpoint constraints ..." << std::endl;
  ct_surface.bezier_midpoint_ind2dep_expanded(
    f2f_expanded, independent_node_map, use_incenter);

  std::cout << "compute interior 2 constraints ..." << std::endl;
  ct_surface.bezier_internal_ind2dep_2_expanded(
    f2f_expanded, independent_node_map, use_incenter);

  std::cout << "done constraint computation" << std::endl;

  // count independent variables
  int64_t dep_cnt = 0;
  int64_t ind_cnt = 0;
  for (int64_t i = 0; i < node_cnt * 3; ++i) {
    if (independent_node_map[i] == 0) {
      dep_cnt++;
    } else if (independent_node_map[i] == 1) {
      ind_cnt++;
    }
  }

  std::cout << "node cnt: " << node_cnt * 3 << std::endl;
  std::cout << "dep cnt: " << dep_cnt << std::endl;
  std::cout << "ind cnt: " << ind_cnt << std::endl;

  // compute constraint matrix, for c1 meshing pipeline, not for cubic
  // optimziation
  Eigen::SparseMatrix<double, 1> bezier_constraint_matrix(dep_cnt,
                                                          node_cnt * 3);
  bezier_constraint_matrix.reserve(Eigen::VectorXi::Constant(dep_cnt, 40));
  int64_t row_id = 0;
  for (size_t i = 0; i < independent_node_map.size(); ++i) {
    if (independent_node_map[i] == 1) {
      // ind, skip
      continue;
    }

    const Eigen::SparseVector<double>& f2f_row = f2f_expanded.row(i);
    assign_spvec_to_spmat_row_help(bezier_constraint_matrix, f2f_row, row_id);
    bezier_constraint_matrix.coeffRef(row_id, i) -= 1;

    row_id++;
  }

  // TODO Make optional
  if (true) {
    Eigen::saveMarket(bezier_constraint_matrix,
                      "CT_bezier_constraints_expanded.txt");
  }

  m_ind2full.resize(node_cnt * 3, ind_cnt);
  m_ind2full.reserve(Eigen::VectorXi::Constant(ind_cnt, 40));
  std::vector<int64_t> col2nid_map;
  std::vector<int64_t> ind2col_map(f2f_expanded.cols(), -1);
  col2nid_map.reserve(f2f_expanded.cols()); // preallocate space
  int64_t col_cnt = 0;
  for (int64_t i = 0; i < f2f_expanded.cols(); ++i) {
    if (independent_node_map[i] == 1) {
      col2nid_map.push_back(i);
      ind2col_map[i] = col_cnt; // map full to independent
      col_cnt++;
    }
  }

  std::ofstream r2f_idx_map_file("CT_bezier_r2f_mat_col_idx_map.txt");
  for (size_t i = 0; i < col2nid_map.size(); ++i) {
    r2f_idx_map_file << col2nid_map[i] << std::endl;
  }
  r2f_idx_map_file.close();

  std::vector<Triplet> ind2full_trips;
  std::vector<bool> diag_seen(f2f_expanded.rows(), false);
  for (int k = 0; k < f2f_expanded.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(
           f2f_expanded, k);
         it;
         ++it) {
      // check if dependent node
      int j = it.col();
      if (independent_node_map[j] == 1) {
        if (ind2col_map[j] < 0)
          spdlog::error("independent column index missing for {}", j);

        // add triplet
        int i = it.row();
        double v = it.value();

        // fix diagonal bug
        if (i == j) {
          if (diag_seen[i]) {
            continue;
          }
          diag_seen[i] = true;
        }

        ind2full_trips.push_back(Triplet(i, ind2col_map[j], v));
      }
    }
  }
  m_ind2full.setFromTriplets(ind2full_trips.begin(), ind2full_trips.end());

  if (true) {
    Eigen::saveMarket(m_ind2full, "CT_bezier_r2f_expanded.txt");
  }

  // build projection from full to independent nodes
  std::vector<Triplet> full2ind_trips;
  for (int i = 0; i < ind_cnt; ++i) {
    int j = col2nid_map[i];
    full2ind_trips.push_back(Triplet(i, j, 1.));
  }
  m_full2ind.resize(ind_cnt, node_cnt * 3);
  m_full2ind.setFromTriplets(full2ind_trips.begin(), full2ind_trips.end());
}

std::array<std::array<int64_t, 10>, 3>
CloughTocherOptimizer::get_micro_triangle_nodes(int64_t face_index) const
{
  const auto& affine_manifold = get_affine_manifold();
  FaceManifoldChart face_chart = affine_manifold.get_face_chart(face_index);
  const auto& l_nodes = face_chart.lagrange_nodes;
  return { { { { l_nodes[0],
                 l_nodes[1],
                 l_nodes[18],
                 l_nodes[3],
                 l_nodes[4],
                 l_nodes[14],
                 l_nodes[15],
                 l_nodes[13],
                 l_nodes[12],
                 l_nodes[9] } },
             { { l_nodes[1],
                 l_nodes[2],
                 l_nodes[18],
                 l_nodes[5],
                 l_nodes[6],
                 l_nodes[16],
                 l_nodes[17],
                 l_nodes[15],
                 l_nodes[14],
                 l_nodes[10] } },
             { { l_nodes[2],
                 l_nodes[0],
                 l_nodes[18],
                 l_nodes[7],
                 l_nodes[8],
                 l_nodes[12],
                 l_nodes[13],
                 l_nodes[17],
                 l_nodes[16],
                 l_nodes[11] } } } };
}

Eigen::SparseMatrix<double>
CloughTocherOptimizer::generate_laplacian_stiffness_matrix() const
{
  // assemble IJV matrix entries
  const auto& affine_manifold = get_affine_manifold();
  std::vector<Triplet> stiffness_matrix_trips;
  int num_faces = affine_manifold.num_faces();
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    std::array<std::array<int64_t, 10>, 3> nodes =
      get_micro_triangle_nodes(fijk);
    assemble_local_laplacian_siffness_matrix(
      face_chart.face_uv_positions, nodes, stiffness_matrix_trips);
  }

  // build matrix
  int node_cnt = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
  Eigen::SparseMatrix<double> stiffness_matrix;
  stiffness_matrix.resize(node_cnt, node_cnt);
  stiffness_matrix.setFromTriplets(stiffness_matrix_trips.begin(),
                                   stiffness_matrix_trips.end());

  return triple_matrix(stiffness_matrix);
}

Eigen::SparseMatrix<double>
CloughTocherOptimizer::generate_laplace_beltrami_stiffness_matrix(
  const std::vector<Eigen::Vector3d>& bezier_control_points) const
{
  // assemble IJV matrix entries
  const auto& affine_manifold = get_affine_manifold();
  std::vector<Triplet> stiffness_matrix_trips;
  int num_faces = affine_manifold.num_faces();
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    std::array<std::array<int64_t, 10>, 3> nodes =
      get_micro_triangle_nodes(fijk);
    assemble_local_laplace_beltrami_siffness_matrix(
      bezier_control_points, nodes, stiffness_matrix_trips);
  }

  // build matrix
  int node_cnt = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
  Eigen::SparseMatrix<double> stiffness_matrix;
  stiffness_matrix.resize(node_cnt, node_cnt);
  stiffness_matrix.setFromTriplets(stiffness_matrix_trips.begin(),
                                   stiffness_matrix_trips.end());

  return triple_matrix(stiffness_matrix);
}

std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
CloughTocherOptimizer::generate_autodiff_laplace_beltrami_stiffness_matrix(
  const std::vector<Eigen::Vector3d>& bezier_control_points) const
{
  // assemble IJV matrix entries
  const auto& affine_manifold = get_affine_manifold();
  int node_cnt = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
  double energy = 0.;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(3 * node_cnt);
  std::vector<Triplet> stiffness_matrix_trips;
  int num_faces = affine_manifold.num_faces();
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    std::array<std::array<int64_t, 10>, 3> nodes =
      get_micro_triangle_nodes(fijk);
    assemble_autodiff_laplace_beltrami_siffness_matrix(
      bezier_control_points, nodes, energy, gradient, stiffness_matrix_trips);
  }

  // build matrix
  Eigen::SparseMatrix<double> stiffness_matrix;
  stiffness_matrix.resize(3 * node_cnt, 3 * node_cnt);
  stiffness_matrix.setFromTriplets(stiffness_matrix_trips.begin(),
                                   stiffness_matrix_trips.end());

  return std::make_tuple(energy, gradient, stiffness_matrix);
}

std::tuple<double, Eigen::VectorXd>
CloughTocherOptimizer::generate_autodiff_laplace_beltrami_gradient(
  const std::vector<Eigen::Vector3d>& bezier_control_points) const
{
  // assemble IJV matrix entries
  const auto& affine_manifold = get_affine_manifold();
  int node_cnt = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
  double energy = 0.;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(3 * node_cnt);
  int num_faces = affine_manifold.num_faces();
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    std::array<std::array<int64_t, 10>, 3> nodes =
      get_micro_triangle_nodes(fijk);
    assemble_autodiff_laplace_beltrami_gradient(
      bezier_control_points, nodes, energy, gradient);
  }

  return std::make_tuple(energy, gradient);
}

// TODO: Obtained from affine_manifold.cpp. Make standalone function
const std::array<PlanarPoint, 19> CT_nodes_uniform = { {
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
} };

Eigen::SparseMatrix<double>
CloughTocherOptimizer::generate_laplace_beltrami_stiffness_matrix() const
{
  // assemble IJV matrix entries
  const auto& affine_manifold = get_affine_manifold();
  std::vector<Triplet> stiffness_matrix_trips;
  int num_faces = affine_manifold.num_faces();
  std::vector<Eigen::Vector3d> bezier_control_points(19);
  Eigen::Matrix<double, 10, 10> p3_lag2bezier_matrix = p3_lag2bezier_m();
  std::array<int64_t, 10> perm = { 0, 9, 3, 4, 7, 8, 6, 2, 1, 5 };
  double cp_3d[10][3];
  double A[10][10];
  std::array<std::array<int64_t, 10>, 3> nodes =
    get_local_micro_triangle_nodes();
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    const auto& P = face_chart.face_uv_positions;
    Eigen::Vector3d Vi = { P[0][0], P[0][1], 0. };
    Eigen::Vector3d Vj = { P[1][0], P[1][1], 0. };
    Eigen::Vector3d Vk = { P[2][0], P[2][1], 0. };
    std::array<std::array<int64_t, 10>, 3> patch_indices =
      get_micro_triangle_nodes(fijk);

    std::array<PlanarPoint, 19> CT_nodes;

    if (m_use_incenter) {
      const double& alpha = face_chart.alpha;
      const double& beta = face_chart.beta;
      const double& gamma = face_chart.gamma;

      CT_nodes = { {
        PlanarPoint(1., 0.),           // b0    0
        PlanarPoint(0., 1.),           // b1    1
        PlanarPoint(0., 0.),           // b2    2
        PlanarPoint(2. / 3., 1. / 3.), // b01   3
        PlanarPoint(1. / 3., 2. / 3.), // b10   4
        PlanarPoint(0., 2. / 3.),      // b12   5
        PlanarPoint(0., 1. / 3.),      // b21   6
        PlanarPoint(1. / 3., 0.),      // b20   7
        PlanarPoint(2. / 3., 0.),      // b02   8
        PlanarPoint(1. / 3. + 1. / 3. * beta,
                    1. / 3. + 1. / 3. * gamma),                 // b01^c 9
        PlanarPoint(1. / 3. * beta, 1. / 3. + 1. / 3. * gamma), // b12^c 10
        PlanarPoint(1. / 3. + 1. / 3. * beta, 1. / 3. * gamma), // b20^c 11
        PlanarPoint(2. / 3. + 1. / 3. * beta, 1. / 3. * gamma), // b0c   12
        PlanarPoint(1. / 3. + 2. / 3. * beta, 2. / 3. * gamma), // bc0   13
        PlanarPoint(1. / 3. * beta, 2. / 3. + 1. / 3. * gamma), // b1c   14
        PlanarPoint(2. / 3. * beta, 1. / 3. + 2. / 3. * gamma), // bc1   15
        PlanarPoint(1. / 3. * beta, 1. / 3. * gamma),           // b2c   16
        PlanarPoint(2. / 3. * beta, 2. / 3. * gamma),           // bc2   17
        PlanarPoint(beta, gamma),                               // bc    18
      } };
    } else {
      CT_nodes = CT_nodes_uniform;
    }

    for (int n = 0; n < 3; ++n) {
      // subtri i
      Eigen::Matrix<double, 10, 3> lag_values_sub, bezier_points_sub;

      for (int k = 0; k < 10; ++k) {
        PlanarPoint uv = CT_nodes[nodes[n][k]];
        double u = uv[0];
        double v = uv[1];
        double w = 1. - u - v;
        lag_values_sub.row(k) = u * Vi + v * Vj + w * Vk;
      }

      // convert local 10 lag to bezier
      bezier_points_sub = p3_lag2bezier_matrix * lag_values_sub;

      // get 3D points
      for (int i = 0; i < 10; i++) {
        for (int d = 0; d < 3; d++) {
          cp_3d[perm[i]][d] = bezier_points_sub(i, d);
        }
      }

      // compute 10x10 local stiffness matrix (same for all dimensions)
      compute_elem_matrix_C(cp_3d, QUAD_DIM, quad_pts, weights, A);

      // build single dimension copy of the stiffness matrix
      for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
          int64_t I = patch_indices[n][i];
          int64_t J = patch_indices[n][j];
          double V = A[perm[i]][perm[j]];
          stiffness_matrix_trips.push_back(Triplet(I, J, V));
        }
      }
    }
  }

  // build matrix
  int node_cnt = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
  Eigen::SparseMatrix<double> stiffness_matrix;
  stiffness_matrix.resize(node_cnt, node_cnt);
  stiffness_matrix.setFromTriplets(stiffness_matrix_trips.begin(),
                                   stiffness_matrix_trips.end());

  return triple_matrix(stiffness_matrix);
}

void
CloughTocherOptimizer::assemble_local_laplace_beltrami_siffness_matrix(
  const std::vector<Eigen::Vector3d>& bezier_control_points,
  const std::array<std::array<int64_t, 10>, 3>& patch_indices,
  std::vector<Triplet>& stiffness_matrix_trips) const
{
  // need to remap from indexing assumed by
  // (a) Lagrange to Bezier conversion to (b) local stiffness matrix
  // enumerate:        0   1   2   3   4   5   6   7   8   9
  // original order: 003 300 030 102 201 210 120 021 012 111
  // new order:      003 012 021 030 102 111 120 201 210 300
  // permutation:      0   9   3   4   7   8   6   2   1   5
  std::array<int64_t, 10> perm = { 0, 9, 3, 4, 7, 8, 6, 2, 1, 5 };

  double cp_3d[10][3];
  double A[10][10];
  for (int n = 0; n < 3; n++) {
    // get 3D points
    for (int i = 0; i < 10; i++) {
      for (int d = 0; d < 3; d++) {
        int64_t I = patch_indices[n][i];
        cp_3d[perm[i]][d] = bezier_control_points[I][d];
      }
    }

    // compute 10x10 local stiffness matrix (same for all dimensions)
    compute_elem_matrix_C(cp_3d, QUAD_DIM, quad_pts, weights, A);

    // build single dimension copy of the stiffness matrix
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        int64_t I = patch_indices[n][i];
        int64_t J = patch_indices[n][j];
        double V = A[perm[i]][perm[j]];
        stiffness_matrix_trips.push_back(Triplet(I, J, V));
      }
    }
  }
}

void
CloughTocherOptimizer::assemble_autodiff_laplace_beltrami_gradient(
  const std::vector<Eigen::Vector3d>& bezier_control_points,
  const std::array<std::array<int64_t, 10>, 3>& patch_indices,
  double& energy,
  Eigen::VectorXd& gradient) const
{
  typedef DScalar1<double, Eigen::Matrix<double, 30, 1>> DiffScalar;
  DiffScalar::setVariableCount(30);

  // need to remap from indexing assumed by
  // (a) Lagrange to Bezier conversion to (b) local stiffness matrix
  // enumerate:        0   1   2   3   4   5   6   7   8   9
  // original order: 003 300 030 102 201 210 120 021 012 111
  // new order:      003 012 021 030 102 111 120 201 210 300
  // permutation:      0   9   3   4   7   8   6   2   1   5
  std::array<int64_t, 10> perm = { 0, 9, 3, 4, 7, 8, 6, 2, 1, 5 };

  DiffScalar cp_3d[10][3];
  DiffScalar A_diff[10][10];
  for (int n = 0; n < 3; n++) {
    // get 3D points
    for (int i = 0; i < 10; i++) {
      for (int d = 0; d < 3; d++) {
        int64_t I = patch_indices[n][i];
        cp_3d[perm[i]][d] = DiffScalar(3 * i + d, bezier_control_points[I][d]);
      }
    }

    // compute 10x10 local stiffness matrix (same for all dimensions)
    compute_elem_matrix_C(cp_3d, QUAD_DIM, quad_pts, weights, A_diff);

    // compute energy as 0.5 x^T A x
    DiffScalar energy_diff(0);
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        for (int d = 0; d < 3; d++) {
          energy_diff += cp_3d[i][d] * cp_3d[j][d] * A_diff[i][j];
        }
        // spdlog::info("local matrix {}, {}: {}", i, j,
        // A_diff[i][j].getValue());
      }
    }
    energy_diff /= 2.;
    energy += energy_diff.getValue();
    Eigen::Matrix<double, 30, 1> g = energy_diff.getGradient();
    // spdlog::info("local matrix:\n{}", A);
    // spdlog::info("local gradient:\n{}", g);

    // build single dimension copy of the stiffness matrix
    for (int i = 0; i < 10; i++) {
      int64_t I = patch_indices[n][i];
      for (int d = 0; d < 3; d++) {
        double gi = g[3 * i + d];
        gradient[3 * I + d] += gi;
      }
    }
  }
}

void
CloughTocherOptimizer::assemble_autodiff_laplace_beltrami_siffness_matrix(
  const std::vector<Eigen::Vector3d>& bezier_control_points,
  const std::array<std::array<int64_t, 10>, 3>& patch_indices,
  double& energy,
  Eigen::VectorXd& gradient,
  std::vector<Triplet>& stiffness_matrix_trips) const
{
  typedef DScalar2<double,
                   Eigen::Matrix<double, 30, 1>,
                   Eigen::Matrix<double, 30, 30>>
    DiffScalar;
  DiffScalar::setVariableCount(30);

  // need to remap from indexing assumed by
  // (a) Lagrange to Bezier conversion to (b) local stiffness matrix
  // enumerate:        0   1   2   3   4   5   6   7   8   9
  // original order: 003 300 030 102 201 210 120 021 012 111
  // new order:      003 012 021 030 102 111 120 201 210 300
  // permutation:      0   9   3   4   7   8   6   2   1   5
  std::array<int64_t, 10> perm = { 0, 9, 3, 4, 7, 8, 6, 2, 1, 5 };

  DiffScalar cp_3d[10][3];
  DiffScalar A_diff[10][10];
  for (int n = 0; n < 3; n++) {
    // get 3D points
    for (int i = 0; i < 10; i++) {
      for (int d = 0; d < 3; d++) {
        int64_t I = patch_indices[n][i];
        cp_3d[perm[i]][d] = DiffScalar(3 * i + d, bezier_control_points[I][d]);
      }
    }

    // compute 10x10 local stiffness matrix (same for all dimensions)
    compute_elem_matrix_C(cp_3d, QUAD_DIM, quad_pts, weights, A_diff);

    // compute energy as 0.5 x^T A x
    DiffScalar energy_diff(0);
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        for (int d = 0; d < 3; d++) {
          energy_diff += cp_3d[i][d] * cp_3d[j][d] * A_diff[i][j];
        }
        // spdlog::info("local matrix {}, {}: {}", i, j,
        // A_diff[i][j].getValue());
      }
    }
    energy_diff /= 2.;
    energy += energy_diff.getValue();
    Eigen::Matrix<double, 30, 1> g = energy_diff.getGradient();
    Eigen::Matrix<double, 30, 30> A = energy_diff.getHessian();
    // spdlog::info("local matrix:\n{}", A);
    // spdlog::info("local gradient:\n{}", g);

    // build single dimension copy of the stiffness matrix
    for (int i = 0; i < 10; i++) {
      int64_t I = patch_indices[n][i];
      for (int d = 0; d < 3; d++) {
        double gi = g[3 * i + d];
        gradient[3 * I + d] += gi;

        for (int j = 0; j < 10; j++) {
          int64_t J = patch_indices[n][j];
          double V = A(3 * i + d, 3 * j + d);
          stiffness_matrix_trips.push_back(Triplet(3 * I + d, 3 * J + d, V));
        }
      }
    }
  }
}

Eigen::SparseMatrix<double>
CloughTocherOptimizer::generate_position_matrix(const Eigen::VectorXd& p) const
{
  // get list of position vertex indices
  const auto& affine_manifold = get_affine_manifold();
  int num_faces = affine_manifold.num_faces();
  int num_nodes = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
  std::vector<bool> is_vertex_node(num_nodes, false);
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    const auto& nodes = face_chart.lagrange_nodes;
    is_vertex_node[nodes[0]] = true;
    is_vertex_node[nodes[1]] = true;
    is_vertex_node[nodes[2]] = true;
  }

  // assemble IJV matrix entries
  std::vector<Triplet> position_matrix_trips;
  for (int i = 0; i < num_nodes; ++i) {
    if (!is_vertex_node[i])
      continue;
    for (int d = 0; d < 3; ++d) {
      int I = 3 * i + d;
      position_matrix_trips.push_back(
        Triplet(I, I, power(std::abs(p[I]), p_norm - 2)));
    }
  }

  // build matrix
  Eigen::SparseMatrix<double> position_matrix;
  position_matrix.resize(3 * num_nodes, 3 * num_nodes);
  position_matrix.setFromTriplets(position_matrix_trips.begin(),
                                  position_matrix_trips.end());

  return position_matrix;
}

Eigen::SparseMatrix<double>
CloughTocherOptimizer::generate_position_matrix() const
{
  // get list of position vertex indices
  const auto& affine_manifold = get_affine_manifold();
  int num_faces = affine_manifold.num_faces();
  int num_nodes = affine_manifold.m_lagrange_nodes.size(); // TODO Replace
  std::vector<bool> is_vertex_node(num_nodes, false);
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    const auto& nodes = face_chart.lagrange_nodes;
    is_vertex_node[nodes[0]] = true;
    is_vertex_node[nodes[1]] = true;
    is_vertex_node[nodes[2]] = true;
  }

  // assemble IJV matrix entries
  std::vector<Triplet> position_matrix_trips;
  for (int i = 0; i < num_nodes; ++i) {
    if (!is_vertex_node[i])
      continue;
    position_matrix_trips.push_back(Triplet(i, i, 1.));
  }

  // build matrix
  Eigen::SparseMatrix<double> position_matrix;
  position_matrix.resize(num_nodes, num_nodes);
  position_matrix.setFromTriplets(position_matrix_trips.begin(),
                                  position_matrix_trips.end());

  return triple_matrix(position_matrix);
}

double
CloughTocherOptimizer::evaluate_quadratic_energy(
  const Eigen::SparseMatrix<double>& H,
  const Eigen::VectorXd& d,
  const double& E0,
  const Eigen::VectorXd& x)
{
  double energy = 0.;
  energy += 0.5 * x.dot(H * x);
  energy += d.dot(x);
  energy += E0;
  return energy;
}

Eigen::SparseMatrix<double>
CloughTocherOptimizer::triple_matrix(
  const Eigen::SparseMatrix<double>& mat) const
{
  int rows = mat.rows();
  std::vector<Triplet> matrix_trips;
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
      int i = it.row();
      int j = it.col();
      double v = it.value();
      for (int n = 0; n < 3; ++n) {
        int I = (i * 3) + n;
        int J = (j * 3) + n;
        matrix_trips.push_back(Triplet(I, J, v));
      }
    }
  }

  Eigen::SparseMatrix<double> tripled_mat;
  tripled_mat.resize(3 * rows, 3 * rows);
  tripled_mat.setFromTriplets(matrix_trips.begin(), matrix_trips.end());

  return tripled_mat;
}

Eigen::VectorXd
CloughTocherOptimizer::build_node_vector(
  const std::vector<Eigen::Vector3d>& bezier_control_points) const
{
  int num_nodes = bezier_control_points.size();
  Eigen::VectorXd p(3 * num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    for (int n = 0; n < 3; ++n) {
      p[(3 * i) + n] = bezier_control_points[i][n];
    }
  }

  return p;
}

std::vector<Eigen::Vector3d>
CloughTocherOptimizer::build_control_points(const Eigen::VectorXd& p) const
{
  int num_nodes = p.size() / 3;
  std::vector<Eigen::Vector3d> bezier_control_points(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    for (int n = 0; n < 3; ++n) {
      bezier_control_points[i][n] = p[(3 * i) + n];
    }
  }

  return bezier_control_points;
}

void
CloughTocherOptimizer::assemble_local_laplacian_siffness_matrix(
  const std::array<PlanarPoint, 3>& face_uv_positions,
  const std::array<std::array<int64_t, 10>, 3>& patch_indices,
  std::vector<Triplet>& stiffness_matrix_trips) const
{
  // extract uv coordinates
  double u[3];
  double v[3];
  for (int n = 0; n < 3; ++n) {
    u[n] = face_uv_positions[n][0];
    v[n] = face_uv_positions[n][1];
  }

  // get subtriangle uv data
  double T[3][2][2];
  compute_UV_2_bary_subtri(u, v, T);

  // get raw matrix data
  double BM[3][3][10][10];
  compute_BM(BM);

  // generate patch matrices
  double AT[3][10][10];
  compute_AT_from_uv(u, v, 0, BM, AT[0]);
  compute_AT_from_uv(u, v, 1, BM, AT[1]);
  compute_AT_from_uv(u, v, 2, BM, AT[2]);

  // compute uv triangle determinants
  double detT[3];
  for (int n = 0; n < 3; n++) {
    detT[n] = T[n][0][0] * T[n][1][1] - T[n][0][1] * T[n][1][0];
  }

  // need to remap from indexing assumed by
  // (a) Lagrange to Bezier conversion to (b) local stiffness matrix
  // enumerate:        0   1   2   3   4   5   6   7   8   9
  // original order: 003 300 030 102 201 210 120 021 012 111
  // new order:      003 012 021 030 102 111 120 201 210 300
  // permutation:      0   9   3   4   7   8   6   2   1   5
  std::array<int64_t, 10> perm = { 0, 9, 3, 4, 7, 8, 6, 2, 1, 5 };

  // assemble local matrix in global matrix
  // WARNING: renormalize by determinant first
  for (int n = 0; n < 3; n++) {
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        int64_t I = patch_indices[n][i];
        int64_t J = patch_indices[n][j];
        double V = AT[n][perm[i]][perm[j]] / detT[n];
        if (double_area)
          V = AT[n][perm[i]][perm[j]];
        stiffness_matrix_trips.push_back(Triplet(I, J, V));
      }
    }
  }
}

void
CloughTocherOptimizer::assemble_patch_coefficients(
  const std::array<int64_t, 10>& patch_indices,
  const CubicHessian& local_hessian,
  std::vector<Triplet>& global_hessian_trips)
{
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      int I = patch_indices[i];
      int J = patch_indices[j];
      double V = local_hessian(i, j);
      global_hessian_trips.push_back(Triplet(I, J, V));
    }
  }
}

std::vector<Eigen::Vector3d>
generate_linear_clough_tocher_surface(CloughTocherSurface& ct_surface,
                                      const Eigen::MatrixXd& V)
{

  // TODO: Obtained from affine_manifold.cpp. Make standalone function
  const std::array<PlanarPoint, 19> CT_nodes = { {
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
  } };

  int num_nodes = ct_surface.m_lagrange_node_values.size();
  std::vector<Eigen::Vector3d> lagrange_control_points(num_nodes);
  const auto& affine_manifold = ct_surface.m_affine_manifold;
  const auto& F = affine_manifold.get_faces();
  int num_faces = affine_manifold.num_faces();
  for (int fijk = 0; fijk < num_faces; ++fijk) {
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    const auto& l_nodes = face_chart.lagrange_nodes;
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

void
set_bezier_control_points(
  CloughTocherSurface& ct_surface,
  const std::vector<Eigen::Vector3d>& bezier_control_points)
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
  std::vector<Eigen::Vector3d> lagrange_control_points(node_cnt);
  for (int64_t i = 0; i < node_cnt; ++i) {
    for (int j = 0; j < 3; ++j) {
      lagrange_control_points[i][j] = lagrange_matrix(i, j);
    }
  }

  const auto& affine_manifold = ct_surface.m_affine_manifold;
  for (int fijk = 0; fijk < affine_manifold.num_faces(); ++fijk) {
    std::array<Eigen::Vector2d, 19> planar_control_points;
    std::array<Eigen::Vector3d, 19> local_control_points;
    FaceManifoldChart face_chart = affine_manifold.get_face_chart(fijk);
    const auto& l_nodes = face_chart.lagrange_nodes;
    for (int i = 0; i < 19; ++i) {
      planar_control_points[i] =
        affine_manifold.m_lagrange_nodes[l_nodes[i]].second;
      local_control_points[i] = lagrange_control_points[l_nodes[i]];
    }
    ct_surface.m_patches[fijk].set_lagrange_nodes(planar_control_points,
                                                  local_control_points);
  }
}

// Helper function to write a curface with external bezier nodes to file
void
write_mesh(CloughTocherSurface& ct_surface,
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

// write edge geometry to file
void
write_polylines_to_obj(const std::string& filename,
                       const std::vector<SpatialVector>& points,
                       const std::vector<std::vector<int>>& polylines)
{
  // write all feature edge vertices
  std::ofstream output_file(filename, std::ios::out | std::ios::trunc);
  int num_points = points.size();
  for (int vi = 0; vi < num_points; ++vi) {
    output_file << "v ";
    for (int i = 0; i < 3; ++i) {
      output_file << std::fixed << std::setprecision(17) << points[vi][i]
                  << " ";
    }
    output_file << std::endl;
  }
  for (const auto& polyline : polylines) {
    int length = polyline.size();
    for (int i = 0; i < length - 1; ++i) {
      int j = (i + 1) % length;
      output_file << "l " << polyline[i] + 1 << " " << polyline[j] + 1
                  << std::endl;
    }
  }
  output_file.close();
}
