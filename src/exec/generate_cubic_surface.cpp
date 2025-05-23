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

void assign_spvec_to_spmat_row_main(Eigen::SparseMatrix<double> &mat,
                                    Eigen::SparseVector<double> &vec,
                                    const int row) {
  for (Eigen::SparseVector<double>::InnerIterator it(vec); it; ++it) {
    mat.coeffRef(row, it.index()) = it.value();
  }
}

void assign_spvec_to_spmat_row_main(Eigen::SparseMatrix<double, 1> &mat,
                                    Eigen::SparseVector<double> &vec,
                                    const int row) {
  for (Eigen::SparseVector<double>::InnerIterator it(vec); it; ++it) {
    mat.coeffRef(row, it.index()) = it.value();
  }
}

void assign_spvec_to_spmat_row_main(Eigen::SparseMatrix<double, 1> &mat,
                                    const Eigen::SparseVector<double> &vec,
                                    const int row) {
  for (Eigen::SparseVector<double>::InnerIterator it(vec); it; ++it) {
    mat.coeffRef(row, it.index()) = it.value();
  }
}

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
  bool skip_constraint = false;
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
  app.add_option("--skip_constraint", skip_constraint,
                 "skip constraint computation if not needed");
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

  // std::cout << "V:" << V << std::endl;
  // std::cout << "F:" << F << std::endl;
  // std::cout << "uv:" << uv << std::endl;
  // std::cout << "FT:" << FT << std::endl;

  // std::cout << "here" << std::endl;
  AffineManifold affine_manifold(F, uv, FT);
  // std::cout << "here2" << std::endl;

  CloughTocherSurface ct_surface(V, affine_manifold, optimization_params,
                                 fit_matrix, energy_hessian,
                                 energy_hessian_inverse);

  std::cout << "#F: " << ct_surface.m_affine_manifold.m_face_charts.size()
            << std::endl;
  std::cout << "#E: " << ct_surface.m_affine_manifold.m_edge_charts.size()
            << std::endl;
  std::cout << "#V: " << ct_surface.m_affine_manifold.m_vertex_charts.size()
            << std::endl;

  // check two-separate satisfaction
  for (size_t vid = 0;
       vid < ct_surface.m_affine_manifold.m_vertex_charts.size(); ++vid) {
    const auto &v_chart = ct_surface.m_affine_manifold.m_vertex_charts[vid];

    if (!v_chart.is_cone) {
      int cone_cnt = 0;
      for (size_t i = 0; i < v_chart.vertex_one_ring.size() - 1; ++i) {
        if (ct_surface.m_affine_manifold
                .m_vertex_charts[v_chart.vertex_one_ring[i]]
                .is_cone) {
          cone_cnt++;
        }
      }

      if (cone_cnt == 0) {
        assert(!v_chart.is_cone_adjacent);
      } else if (cone_cnt == 1) {
        assert(v_chart.is_cone_adjacent);
      } else if (cone_cnt > 1) {
        // assert(false);
        throw std::runtime_error(
            "non-cone vertex is adjacent to more than one cone, cannot setup "
            "cone constraint! Try moving the cones or denser meshes!");
      }
    }
  }

  // important !!! call the following two before compute constraints, but only
  // once!!!
  ct_surface.m_affine_manifold.generate_lagrange_nodes();
  ct_surface.m_affine_manifold.compute_edge_global_uv_mappings();

  ct_surface.write_cubic_surface_to_msh_with_conn_from_lagrange_nodes(
      output_name + "_from_lagrange_nodes");
  ct_surface.write_cubic_surface_to_msh_with_conn_from_lagrange_nodes(
      output_name + "_from_bezier_nodes", true);

  ct_surface.write_connected_lagrange_nodes(output_name + "_bilaplacian_nodes",
                                            V);
  ct_surface.write_connected_lagrange_nodes_values(output_name +
                                                   "_bilaplacian_nodes_values");

  Eigen::SparseMatrix<double, 1> b2l_mat;
  ct_surface.bezier2lag_full_mat(b2l_mat);

  Eigen::saveMarket(b2l_mat,
                    output_name + "_bezier_to_lag_convertion_matrix.txt");

  if (skip_constraint) {
    // skip constraint computation
    exit(0);
  }

  // get vertex normals
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

  //////////////////////////////////
  /////     bezier form    /////////
  //////////////////////////////////

  // int64_t node_cnt = ct_surface.m_affine_manifold.m_lagrange_nodes.size();
  // Eigen::SparseMatrix<double> bezier_full_to_full(node_cnt, node_cnt);

  // std::vector<int64_t> constrained_row_ids;
  // std::map<int64_t, int> independent_node_map;
  // ct_surface.Ci_endpoint_ind2dep(bezier_full_to_full, constrained_row_ids,
  //                                independent_node_map);
  // // Eigen::saveMarket(bezier_full_to_full, output_name +
  // // "_bezier_1_r2f.txt");
  // ct_surface.Ci_internal_ind2dep_1(bezier_full_to_full, constrained_row_ids,
  //                                  independent_node_map);
  // // Eigen::saveMarket(bezier_full_to_full, output_name +
  // // "_bezier_12_r2f.txt");
  // ct_surface.Ci_midpoint_ind2dep(bezier_full_to_full, constrained_row_ids,
  //                                independent_node_map);
  // ct_surface.Ci_internal_ind2dep_2(bezier_full_to_full, constrained_row_ids,
  //                                  independent_node_map);

  // std::cout << "constrained row count: " << constrained_row_ids.size()
  //           << std::endl;
  // // Eigen::saveMarket(bezier_full_to_full, output_name + "_bezier_r2f.txt");

  // Eigen::SparseMatrix<double> bezier_cone_matrix;
  // ct_surface.Ci_cone_bezier(bezier_full_to_full, bezier_cone_matrix,
  // v_normals);

  // Eigen::saveMarket(bezier_cone_matrix, output_name + "_bezier_cone.txt");

  // // check bezier constraint
  // auto bezier_control_points = ct_surface.m_bezier_control_points;
  // Eigen::MatrixXd bezier_points_mat(bezier_control_points.size(), 3);
  // for (size_t i = 0; i < bezier_control_points.size(); ++i) {
  //   bezier_points_mat.row(i) = bezier_control_points[i].transpose();
  // }

  // // bezier_points_mat.setConstant(1);

  // // extract constrained lines
  // Eigen::SparseMatrix<double>
  // bezier_endpoint_cons(constrained_row_ids.size(),
  //                                                  node_cnt);

  // Eigen::MatrixXd endpoint_bezier_points_mat;
  // endpoint_bezier_points_mat.resize(constrained_row_ids.size(), 3);
  // for (size_t i = 0; i < constrained_row_ids.size(); ++i) {
  //   Eigen::SparseVector<double> tmpvec =
  //       bezier_full_to_full.row(constrained_row_ids[i]);
  //   assign_spvec_to_spmat_row_main(bezier_endpoint_cons, tmpvec, i);

  //   endpoint_bezier_points_mat.row(i) =
  //       bezier_points_mat.row(constrained_row_ids[i]);
  // }

  // auto beizer_end_error =
  //     bezier_endpoint_cons * bezier_points_mat - endpoint_bezier_points_mat;

  // // auto beizer_end_error =
  // //     bezier_full_to_full * bezier_points_mat - bezier_points_mat;
  // double bezier_end_max_error = beizer_end_error.maxCoeff();
  // double bezier_end_min_error = beizer_end_error.minCoeff();

  // std::cout << "beizer max error: "
  //           << ((std::abs(bezier_end_max_error) >
  //                std::abs(bezier_end_min_error))
  //                   ? std::abs(bezier_end_max_error)
  //                   : std::abs(bezier_end_min_error))
  //           << std::endl;

  // // std::cout << beizer_end_error << std::endl;

  // // std::cout << bezier_endpoint_cons * bezier_points_mat << std::endl;

  // // std::cout << std::endl << endpoint_bezier_points_mat << std::endl;

  // std::cout << constrained_row_ids.size() << "  " << node_cnt << std::endl;

  // // compute bezier constraint matrix without cone cons
  // Eigen::SparseMatrix<double> bezier_cons_no_cone;

  // // count dep rows
  // int64_t dep_cnt = 0;
  // int64_t ind_cnt = 0;
  // for (size_t i = 0; i < constrained_row_ids.size(); ++i) {
  //   if (independent_node_map[i] == 0) {
  //     dep_cnt++;
  //   } else if (independent_node_map[i] == 1) {
  //     ind_cnt++;
  //   }
  // }

  // assert(dep_cnt + ind_cnt == node_cnt);

  // bezier_cons_no_cone.resize(dep_cnt, node_cnt);

  // int64_t row_id = 0;
  // for (size_t i = 0; i < constrained_row_ids.size(); ++i) {
  //   if (independent_node_map[i] == 1) {
  //     // skip indepdent nodes
  //     continue;
  //   }

  //   const Eigen::SparseVector<double> &r2f_row = bezier_full_to_full.row(i);
  //   Eigen::SparseVector<double> eye_row;
  //   eye_row.resize(node_cnt);
  //   eye_row.insert(i) = 1;

  //   Eigen::SparseVector<double> cons_row = r2f_row - eye_row;
  //   assign_spvec_to_spmat_row_main(bezier_cons_no_cone, cons_row, row_id);
  //   row_id++;
  // }

  // Eigen::saveMarket(bezier_cons_no_cone,
  //                   output_name + "_bezier_constraints_no_cone.txt");

  // // compute reduce to full without cones
  // Eigen::SparseMatrix<double> bezier_no_cone_r2f(node_cnt, ind_cnt);
  // std::vector<int64_t> col2nodeid_map;
  // int64_t col_cnt = 0;
  // for (int64_t i = 0; i < bezier_full_to_full.cols(); ++i) {
  //   if (independent_node_map[i] == 1) {
  //     const Eigen::SparseVector<double> &c = bezier_full_to_full.col(i);
  //     bezier_no_cone_r2f.col(col_cnt) = c;
  //     col_cnt++;
  //     col2nodeid_map.push_back(i);
  //   } else {
  //     // do nothing
  //   }
  // }

  // std::cout << "ind node cnt: " << ind_cnt << std::endl;

  // Eigen::saveMarket(bezier_no_cone_r2f,
  //                   output_name + "_bezier_r2f_no_cone.txt");

  // std::ofstream r2f_idx_map_file(output_name +
  //                                "_bezier_r2f_mat_col_idx_map.txt");
  // for (size_t i = 0; i < col2nodeid_map.size(); ++i) {
  //   r2f_idx_map_file << col2nodeid_map[i] << std::endl;
  // }
  // r2f_idx_map_file.close();

  // // test bezier lag convertion
  // Eigen::SparseMatrix<double> b2l_full_mat, l2b_full_mat;
  // ct_surface.bezier2lag_full_mat(b2l_full_mat);
  // ct_surface.lag2bezier_full_mat(l2b_full_mat);

  // Eigen::MatrixXd lag_mat(ct_surface.m_lagrange_node_values.size(), 3);
  // for (size_t i = 0; i < ct_surface.m_lagrange_node_values.size(); ++i) {
  //   lag_mat.row(i) = ct_surface.m_lagrange_node_values[i].transpose();
  // }

  // Eigen::MatrixXd bezier_mat(ct_surface.m_bezier_control_points.size(), 3);
  // for (size_t i = 0; i < ct_surface.m_bezier_control_points.size(); ++i) {
  //   bezier_mat.row(i) = ct_surface.m_bezier_control_points[i].transpose();
  // }

  // std::cout << (b2l_full_mat * bezier_mat - lag_mat).norm() << std::endl;
  // std::cout << (l2b_full_mat * lag_mat - bezier_mat).norm() << std::endl;

  // Eigen::saveMarket(b2l_full_mat,
  //                   output_name + "_bezier_to_lag_convertion_matrix.txt");

  // Eigen::saveMarket(l2b_full_mat,
  //                   output_name + "_lag_to_bezier_convertion_matrix.txt");

  // exit(0);

  ////////////////////////////////////////////////////
  /////     bezier form  expanded with cone  /////////
  ////////////////////////////////////////////////////

  int64_t node_cnt = ct_surface.m_affine_manifold.m_lagrange_nodes.size();
  Eigen::SparseMatrix<double, Eigen::RowMajor> f2f_expanded(node_cnt * 3,
                                                            node_cnt * 3);
  // f2f_expanded.reserve(node_cnt * 3 * 5);
  f2f_expanded.reserve(Eigen::VectorXi::Constant(node_cnt * 3, 40));

  // std::map<int64_t, int> independent_node_map;
  std::vector<int> independent_node_map(node_cnt * 3, -1);
  // for (int64_t i = 0; i < node_cnt * 3; ++i) {
  //   // init to -1
  //   independent_node_map[i] = -1;
  // }

  std::vector<bool> node_assigned(node_cnt, false);

  // std::cout << "compute cone constraints ..." << std::endl;
  // ct_surface.bezier_cone_constraints_expanded(
  //     f2f_expanded, independent_node_map, node_assigned, v_normals);

  std::cout << "compute endpoint constraints ..." << std::endl;
  // ct_surface.bezier_endpoint_ind2dep_expanded(f2f_expanded,
  //                                             independent_node_map, false);
  ct_surface.bezier_endpoint_ind2dep_expanded(f2f_expanded,
                                              independent_node_map, true);

  std::cout << "compute interior 1 constraints ..." << std::endl;
  ct_surface.bezier_internal_ind2dep_1_expanded(f2f_expanded,
                                                independent_node_map);

  std::cout << "compute midpoint constraints ..." << std::endl;
  ct_surface.bezier_midpoint_ind2dep_expanded(f2f_expanded,
                                              independent_node_map);

  std::cout << "compute interior 2 constraints ..." << std::endl;
  ct_surface.bezier_internal_ind2dep_2_expanded(f2f_expanded,
                                                independent_node_map);

  std::cout << "done constraint computation" << std::endl;

  int64_t ind_cnt = 0;
  int64_t dep_cnt = 0;
  for (int64_t i = 0; i < node_cnt * 3; ++i) {
    // init to -1
    if (independent_node_map[i] < 0) {
      // std::cout << "node id: " << i << " not processed" << std::endl;
    } else if (independent_node_map[i] == 0) {
      dep_cnt++;
    } else if (independent_node_map[i] == 1) {
      ind_cnt++;
    }
  }

  std::cout << "node cnt: " << node_cnt * 3 << std::endl;
  std::cout << "dep cnt: " << dep_cnt << std::endl;
  std::cout << "ind cnt: " << ind_cnt << std::endl;

  int64_t cone_valance = 0;
  int64_t cone_cnt = 0;
  for (const auto &v_chart : ct_surface.m_affine_manifold.m_vertex_charts) {
    if (v_chart.is_cone) {
      cone_valance += (v_chart.vertex_one_ring.size() - 1);
      cone_cnt++;
    }
  }
  std::cout << "cone total valance: " << cone_valance << std::endl;

  int64_t ind_target_cnt =
      ct_surface.m_affine_manifold.m_vertex_charts.size() * 3 * 3 +
      ct_surface.m_affine_manifold.m_edge_charts.size() * 1 * 3 -
      cone_valance * 2 - cone_cnt * 2 * 3;

  std::cout << "ind target cnt: " << ind_target_cnt << std::endl;

  Eigen::SparseMatrix<double, 1> bezier_constraint_matrix(dep_cnt,
                                                          node_cnt * 3);
  bezier_constraint_matrix.reserve(Eigen::VectorXi::Constant(dep_cnt, 40));
  int64_t row_id = 0;
  for (size_t i = 0; i < independent_node_map.size(); ++i) {
    if (independent_node_map[i] == 1) {
      // ind, skip
      continue;
    }

    const Eigen::SparseVector<double> &f2f_row = f2f_expanded.row(i);
    assign_spvec_to_spmat_row_main(bezier_constraint_matrix, f2f_row, row_id);
    bezier_constraint_matrix.coeffRef(row_id, i) -= 1;

    // Eigen::SparseVector<double> eye_row;
    // eye_row.resize(node_cnt * 3);
    // eye_row.insert(i) = 1;

    // Eigen::SparseVector<double> cons_row = f2f_row - eye_row;
    // assign_spvec_to_spmat_row_main(bezier_constraint_matrix, cons_row,
    // row_id);
    row_id++;
  }

  Eigen::saveMarket(bezier_constraint_matrix,
                    output_name + "_bezier_constraints_expanded_old.txt");

  // generate degenerated bezier control points
  ct_surface.compute_degenerate_bezier_control_points(
      f2f_expanded, independent_node_map, V, F);
  ct_surface.write_degenerate_cubic_surface_to_msh_with_conn(
      "degenerated_c1_bezier_control_points");

  const auto error_degen =
      bezier_constraint_matrix *
      ct_surface.m_degenerated_bezier_control_points_expanded;
  std::cout << "error degen mesh: " << error_degen.norm() << std::endl;

  // Eigen::MatrixXd degenrated_bc(
  //     ct_surface.m_degenerated_bezier_control_points.size(), 3);
  // for (size_t i = 0; i <
  // ct_surface.m_degenerated_bezier_control_points.size();
  //      ++i) {
  //   degenrated_bc.row(i) = ct_surface.m_degenerated_bezier_control_points[i];
  // }

  // Eigen::MatrixXd degenerated_lag = b2l_mat * degenrated_bc;

  // exit(0);

  // compute reduce to full
  std::cout << "computing reduce to full" << std::endl;
  Eigen::SparseMatrix<double> r2f_expanded(node_cnt * 3, ind_cnt);
  // r2f_expanded.reserve(node_cnt * 3 * 5);
  r2f_expanded.reserve(Eigen::VectorXi::Constant(ind_cnt, 40));
  std::vector<int64_t> col2nid_map;
  std::vector<int64_t> ind2col_map(f2f_expanded.cols(), -1);
  int64_t col_cnt = 0;
  for (int64_t i = 0; i < f2f_expanded.cols(); ++i) {
    if (independent_node_map[i] == 1) {
      // const Eigen::SparseVector<double> &c = f2f_expanded.col(i);
      // r2f_expanded.col(col_cnt) = c;
      col2nid_map.push_back(i);
      ind2col_map[i] = col_cnt; // map full to independent
      col_cnt++;
    }
  }

  std::ofstream r2f_idx_map_file(output_name +
                                 "_bezier_r2f_mat_col_idx_map_old.txt");
  for (size_t i = 0; i < col2nid_map.size(); ++i) {
    r2f_idx_map_file << col2nid_map[i] << std::endl;
  }
  r2f_idx_map_file.close();

  std::vector<Eigen::Triplet<double>> ind2full_trips;
  std::vector<bool> diag_seen(f2f_expanded.rows(), false);
  for (int k = 0; k < f2f_expanded.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(
             f2f_expanded, k);
         it; ++it) {
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

        ind2full_trips.push_back(Eigen::Triplet<double>(i, ind2col_map[j], v));
      }
    }
  }
  r2f_expanded.setFromTriplets(ind2full_trips.begin(), ind2full_trips.end());

  Eigen::saveMarket(r2f_expanded, output_name + "_bezier_r2f_expanded_old.txt");

  exit(0);

  // compute expanded beizer control points
  std::cout << "computing bezier c points expanded" << std::endl;
  const auto &bezier_points = ct_surface.m_bezier_control_points;
  Eigen::VectorXd bezier_points_expanded(bezier_points.size() * 3);

  for (size_t i = 0; i < bezier_points.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      bezier_points_expanded[i * 3 + j] = bezier_points[i][j];
    }
  }

  // extract ind bezier control points
  Eigen::VectorXd bezier_ind(ind_cnt);
  for (int64_t i = 0; i < ind_cnt; ++i) {
    bezier_ind[i] = bezier_points_expanded[col2nid_map[i]];
  }

  // compute contraint bezier cp
  Eigen::VectorXd full_from_cone_expanded = r2f_expanded * bezier_ind;
  Eigen::MatrixXd full_from_cone(node_cnt, 3);
  for (int64_t i = 0; i < node_cnt; ++i) {
    for (int j = 0; j < 3; ++j) {
      full_from_cone(i, j) = full_from_cone_expanded[i * 3 + j];
    }
  }

  // Eigen::SparseMatrix<double> b2l_mat;
  // ct_surface.bezier2lag_full_mat(b2l_mat);

  Eigen::MatrixXd full_from_cone_lag = b2l_mat * full_from_cone;

  ct_surface.write_external_point_values_with_conn("cone_cons_lag_nodes_mesh",
                                                   full_from_cone_lag);
  ct_surface.write_external_point_values_with_conn(
      "cone_cons_bezier_nodes_mesh", full_from_cone);

  Eigen::SparseMatrix<double> old_cons(node_cnt, node_cnt);
  std::vector<int64_t> old_constrained_row_ids;
  std::map<int64_t, int> old_independent_node_map;
  ct_surface.Ci_endpoint_ind2dep(old_cons, old_constrained_row_ids,
                                 old_independent_node_map);
  ct_surface.Ci_internal_ind2dep_1(old_cons, old_constrained_row_ids,
                                   old_independent_node_map);
  ct_surface.Ci_midpoint_ind2dep(old_cons, old_constrained_row_ids,
                                 old_independent_node_map);
  ct_surface.Ci_internal_ind2dep_2(old_cons, old_constrained_row_ids,
                                   old_independent_node_map);

  Eigen::SparseMatrix<double> eye(node_cnt, node_cnt);
  for (int64_t i = 0; i < node_cnt; ++i) {
    eye.insert(i, i) = 1;
  }

  Eigen::MatrixXd error_mat = (old_cons - eye) * full_from_cone;
  double max_error = error_mat.maxCoeff();
  double min_error = error_mat.minCoeff();
  std::cout << "error from old cons max: " << max_error << std::endl;
  std::cout << "error from old cons min: " << min_error << std::endl;

  // Eigen::saveMarket(bezier_constraint_matrix,
  //                   output_name + "_bezier_constraints_expanded.txt");

  exit(0);

  ////////////////////////////////////////////////////////
  // lagrange form
  ////////////////////////////////////////////////////////

  Eigen::SparseMatrix<double> c_f_int;
  ct_surface.C_F_int(c_f_int);
  Eigen::SparseMatrix<double> C_e_end, C_e_end_elim;
  ct_surface.C_E_end(C_e_end, C_e_end_elim);
  Eigen::SparseMatrix<double> C_e_mid;
  ct_surface.C_E_mid(C_e_mid);

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
