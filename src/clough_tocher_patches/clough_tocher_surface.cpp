#include "clough_tocher_surface.hpp"

#include <fstream>
#include <igl/Timer.h>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>
#include <unsupported/Eigen/SparseExtra>

#include "clough_tocher_constraint_matrices.hpp"

CloughTocherSurface::CloughTocherSurface() {}

CloughTocherSurface::CloughTocherSurface(
    const Eigen::MatrixXd &V, const AffineManifold &affine_manifold,
    const OptimizationParameters &optimization_params,
    Eigen::SparseMatrix<double> &fit_matrix,
    Eigen::SparseMatrix<double> &energy_hessian,
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
        &energy_hessian_inverse)
    : m_affine_manifold(affine_manifold) {

  // Generate normals
  MatrixXr N;
  generate_face_normals(V, affine_manifold, N);

  // Generate fit matrix by setting the parametrized quadratic surface mapping
  // factor to zero
  double fit_energy;
  VectorXr fit_derivatives;
  OptimizationParameters optimization_params_fit = optimization_params;
  optimization_params_fit.parametrized_quadratic_surface_mapping_factor = 0.0;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> fit_matrix_inverse;
  build_twelve_split_spline_energy_system(
      V, N, affine_manifold, optimization_params_fit, fit_energy,
      fit_derivatives, fit_matrix, fit_matrix_inverse);

  // Build full energy hessian system
  double energy;
  VectorXr derivatives;
  build_twelve_split_spline_energy_system(
      V, N, affine_manifold, optimization_params, energy, derivatives,
      energy_hessian, energy_hessian_inverse);

  // Build optimized corner and midpoint data
  generate_optimized_twelve_split_position_data(V, affine_manifold, fit_matrix,
                                                energy_hessian_inverse,
                                                m_corner_data, m_midpoint_data);

  // compute patches
  assert(m_corner_data.size() == m_midpoint_data.size());
  for (size_t i = 0; i < m_corner_data.size(); ++i) {
    // get cubic data
    Eigen::Matrix<double, 12, 3> boundary_data;
    Eigen::Matrix<double, 1, 3> p0, p1, p2, d01, d10, d12, d21, d20, d02;

    p0 = m_corner_data[i][0].function_value;          // p0
    p1 = m_corner_data[i][1].function_value;          // p1
    p2 = m_corner_data[i][2].function_value;          // p2
    d01 = m_corner_data[i][0].first_edge_derivative;  // d01
    d10 = m_corner_data[i][1].second_edge_derivative; // d10
    d12 = m_corner_data[i][1].first_edge_derivative;  // d12
    d21 = m_corner_data[i][2].second_edge_derivative; // d21
    d20 = m_corner_data[i][2].first_edge_derivative;  // d20
    d02 = m_corner_data[i][0].second_edge_derivative; // d02

    // fix hij
    // convert hij from quadratic to cubic
    Eigen::Matrix<double, 1, 3> h01_q = m_midpoint_data[i][2].normal_derivative;
    Eigen::Matrix<double, 1, 3> h12_q = m_midpoint_data[i][0].normal_derivative;
    Eigen::Matrix<double, 1, 3> h20_q = m_midpoint_data[i][1].normal_derivative;
    Eigen::Matrix<double, 1, 3> h01_c, h12_c, h20_c;

    // formulas
    // dfde_quad = 2*(fj-fi) + 0.5*(dji-dij)
    // dfde_cubic = 1.5*(fj-fi) + 0.25*(dji-dij)
    // hij_fixed = hij - ([df/de]_quad (eij dot mij) + ([df/de]_cubic (eijdot
    // mij)

    // fix edge 01
    const auto &e01_chart = m_affine_manifold.get_edge_chart(i, 2);
    const auto &m01 =
        (e01_chart.top_face_index == int64_t(i))
            ? e01_chart.top_vertex_uv_position
            : e01_chart
                  .bottom_vertex_uv_position; // vector start point is (0,0)
    const auto e01 = (e01_chart.top_face_index == int64_t(i))
                         ? Eigen::Vector2d(1, 0)
                         : Eigen::Vector2d(-1, 0);
    const auto dfde_01_q = 2 * (p1 - p0) + 0.5 * (d10 - d01);
    const auto dfde_01_c = 1.5 * (p1 - p0) + 0.25 * (d10 - d01);
    h01_c = h01_q - dfde_01_q * (e01.dot(m01)) + dfde_01_c * (e01.dot(m01));

    // fix edge 12
    const auto &e12_chart = m_affine_manifold.get_edge_chart(i, 0);
    const auto &m12 =
        (e12_chart.top_face_index == int64_t(i))
            ? e12_chart.top_vertex_uv_position
            : e12_chart
                  .bottom_vertex_uv_position; // vector start point is (0,0)
    const auto e12 = (e12_chart.top_face_index == int64_t(i))
                         ? Eigen::Vector2d(1, 0)
                         : Eigen::Vector2d(-1, 0);
    const auto dfde_12_q = 2 * (p2 - p1) + 0.5 * (d21 - d12);
    const auto dfde_12_c = 1.5 * (p2 - p1) + 0.25 * (d21 - d12);
    h12_c = h12_q - dfde_12_q * (e12.dot(m12)) + dfde_12_c * (e12.dot(m12));

    // fix edge 20
    const auto &e20_chart = m_affine_manifold.get_edge_chart(i, 1);
    const auto &m20 =
        (e20_chart.top_face_index == int64_t(i))
            ? e20_chart.top_vertex_uv_position
            : e20_chart
                  .bottom_vertex_uv_position; // vector start point is (0,0)
    const auto e20 = (e20_chart.top_face_index == int64_t(i))
                         ? Eigen::Vector2d(1, 0)
                         : Eigen::Vector2d(-1, 0);
    const auto dfde_20_q = 2 * (p0 - p2) + 0.5 * (d02 - d20);
    const auto dfde_20_c = 1.5 * (p0 - p2) + 0.25 * (d02 - d20);
    h20_c = h20_q - dfde_20_q * (e20.dot(m20)) + dfde_20_c * (e20.dot(m20));

    boundary_data.row(0) = p0;
    boundary_data.row(1) = p1;
    boundary_data.row(2) = p2;
    boundary_data.row(3) = d01;
    boundary_data.row(4) = d10;
    boundary_data.row(5) = d12;
    boundary_data.row(6) = d21;
    boundary_data.row(7) = d20;
    boundary_data.row(8) = d02;
    boundary_data.row(9) = h01_c;
    boundary_data.row(10) = h12_c;
    boundary_data.row(11) = h20_c;

    // create patch
    m_patches.push_back(CloughTocherPatch(boundary_data));
  }
}

Eigen::Matrix<double, 1, 3>
CloughTocherSurface::evaluate_patch(const PatchIndex &patch_index,
                                    const double &u, const double &v) {
  return m_patches[patch_index].CT_eval(u, v);
}

void CloughTocherSurface::generate_face_normals(
    const Eigen::MatrixXd &V, const AffineManifold &affine_manifold,
    Eigen::MatrixXd &N) {
  Eigen::MatrixXi const &F = affine_manifold.get_faces();

  // Compute the cones of the affine manifold
  std::vector<AffineManifold::Index> cones;
  affine_manifold.compute_cones(cones);

  std::cout << "#Cone: " << cones.size() << std::endl;

  // Get vertex normals
  Eigen::MatrixXd N_vertices;
  igl::per_vertex_normals(V, F, N_vertices);

  // Set the face one ring normals of the cone vertices to the cone vertex
  // normal
  N.setZero(F.rows(), 3);
  for (size_t i = 0; i < cones.size(); ++i) {
    int ci = cones[i];
    VertexManifoldChart const &chart = affine_manifold.get_vertex_chart(ci);
    for (size_t j = 0; j < chart.face_one_ring.size(); ++j) {
      int fj = chart.face_one_ring[j];
      N.row(fj) = N_vertices.row(ci);
    }
  }
}

void CloughTocherSurface::write_cubic_surface_to_msh_no_conn(
    std::string filename) {
  std::ofstream file(filename);

  /*
    $MeshFormat
      4.1 0 8     MSH4.1, ASCII
      $EndMeshFormat
    */

  file << "$MeshFormat\n"
       << "4.1 0 8\n"
       << "$EndMeshFormat\n";

  // msh 10 node 3rd-order triangle nodes
  const std::array<PlanarPoint, 10> tri_0_bcs = {
      {PlanarPoint(0, 0), PlanarPoint(1., 0), PlanarPoint(1. / 3., 1. / 3.),
       PlanarPoint(1. / 3., 0), PlanarPoint(2. / 3., 0),
       PlanarPoint(7. / 9., 1. / 9.), PlanarPoint(5. / 9., 2. / 9.),
       PlanarPoint(2. / 9., 2. / 9.), PlanarPoint(1. / 9., 1. / 9.),
       PlanarPoint(4. / 9., 1. / 9.)}};

  const std::array<PlanarPoint, 10> tri_1_bcs = {
      {PlanarPoint(1, 0), PlanarPoint(0, 1), PlanarPoint(1. / 3., 1. / 3.),
       PlanarPoint(2. / 3., 1. / 3.), PlanarPoint(1. / 3., 2. / 3.),
       PlanarPoint(1. / 9., 7. / 9.), PlanarPoint(2. / 9., 5. / 9.),
       PlanarPoint(5. / 9., 2. / 9.), PlanarPoint(7. / 9., 1. / 9.),
       PlanarPoint(4. / 9., 4. / 9.)}};

  const std::array<PlanarPoint, 10> tri_2_bcs = {
      {PlanarPoint(0, 1), PlanarPoint(0, 0), PlanarPoint(1. / 3., 1. / 3.),
       PlanarPoint(0, 2. / 3.), PlanarPoint(0, 1. / 3.),
       PlanarPoint(1. / 9., 1. / 9.), PlanarPoint(2. / 9., 2. / 9.),
       PlanarPoint(2. / 9., 5. / 9.), PlanarPoint(1. / 9., 7. / 9.),
       PlanarPoint(1. / 9., 4. / 9.)}};

  file << "$Nodes\n";

  const size_t node_size = m_patches.size() * 30;
  file << "1 " << node_size << " 1 " << node_size << "\n";
  file << "2 1 0 " << node_size << "\n";

  for (size_t i = 1; i <= node_size; ++i) {
    file << i << "\n";
  }

  for (const auto &patch : m_patches) {
    for (const auto &bc : tri_0_bcs) {
      auto z = patch.CT_eval(bc[0], bc[1]);
      file << z(0, 0) << " " << z(0, 1) << " " << z(0, 2) << "\n";
    }
    for (const auto &bc : tri_1_bcs) {
      auto z = patch.CT_eval(bc[0], bc[1]);
      file << z(0, 0) << " " << z(0, 1) << " " << z(0, 2) << "\n";
    }
    for (const auto &bc : tri_2_bcs) {
      auto z = patch.CT_eval(bc[0], bc[1]);
      file << z(0, 0) << " " << z(0, 1) << " " << z(0, 2) << "\n";
    }
  }

  file << "$EndNodes\n";

  // write elements
  const size_t element_size = m_patches.size() * 3;

  file << "$Elements\n";
  file << "1 " << element_size << " 1 " << element_size << "\n";
  file << "2 1 21 " << element_size << "\n";
  for (size_t i = 0; i < element_size; ++i) {
    file << i + 1 << " " << i * 10 + 1 << " " << i * 10 + 2 << " " << i * 10 + 3
         << " " << i * 10 + 4 << " " << i * 10 + 5 << " " << i * 10 + 6 << " "
         << i * 10 + 7 << " " << i * 10 + 8 << " " << i * 10 + 9 << " "
         << i * 10 + 10 << "\n";
  }

  file << "$EndElements\n";
}

void CloughTocherSurface::write_coeffs_to_obj(std::string filename) {
  std::ofstream file(filename);

  for (const auto &patch : m_patches) {
    std::array<Eigen::Matrix<double, 10, 3>, 3> coeffs = patch.get_coeffs();
    for (int i = 0; i < 1; ++i) {
      for (int j = 0; j < 10; ++j) {
        file << "v " << coeffs[i](j, 0) << " " << coeffs[i](j, 1) << " "
             << coeffs[i](j, 2) << "\n";
      }
    }
  }

  file << "f 1 1 1\n";
}

void CloughTocherSurface::sample_to_obj(std::string filename, int sample_size) {
  std::ofstream file(filename);

  for (const auto &patch : m_patches) {
    for (int i = 0; i <= sample_size; ++i) {
      for (int j = 0; j <= sample_size - i; ++j) {
        double u = 1. / sample_size * i;
        double v = 1. / sample_size * j;
        // double w = 1 - u - v;

        // std::cout << "u: " << u << " v: " << v << " w: " << w << std::endl;
        // std::cout << "w^3: " << w * w * w << std::endl;
        auto z = patch.CT_eval(u, v);

        // std::cout << z << std::endl;

        file << "v " << z[0] << " " << z[1] << " " << z[2] << '\n';
      }
    }
  }

  file << "f 1 1 1\n";
}

// deprecated
void CloughTocherSurface::write_cubic_surface_to_msh_with_conn(
    std::string filename) {
  std::ofstream file(filename + ".msh");

  /*
    $MeshFormat
      4.1 0 8     MSH4.1, ASCII
      $EndMeshFormat
    */

  file << "$MeshFormat\n"
       << "4.1 0 8\n"
       << "$EndMeshFormat\n";

  /*
  subtri0: 0 4 5 1 12 13 3 11 10 16
  subtri1: 1 6 7 2 14 15 3 13 12 17
  subtri2: 2 8 9 0 10 11 3 15 14 18
  */

  const std::array<PlanarPoint, 19> CT_nodes = {{
      PlanarPoint(0., 0.),           PlanarPoint(1., 0.),
      PlanarPoint(0., 1.),           PlanarPoint(1. / 3., 1. / 3.),
      PlanarPoint(1. / 3., 0.),      PlanarPoint(2. / 3., 0.),
      PlanarPoint(2. / 3., 1. / 3.), PlanarPoint(1. / 3., 2. / 3.),
      PlanarPoint(0., 2. / 3.),      PlanarPoint(0., 1. / 3.),
      PlanarPoint(1. / 9., 1. / 9.), PlanarPoint(2. / 9., 2. / 9.),
      PlanarPoint(7. / 9., 1. / 9.), PlanarPoint(5. / 9., 2. / 9.),
      PlanarPoint(1. / 9., 7. / 9.), PlanarPoint(2. / 9., 5. / 9.),
      PlanarPoint(4. / 9., 1. / 9.), PlanarPoint(4. / 9., 4. / 9.),
      PlanarPoint(1. / 9., 4. / 9.),
  }};

  // std::vector<Eigen::Vector3d> vertices;
  std::map<std::pair<int64_t, int64_t>, std::array<int64_t, 4>>
      boundary_edge_to_v_map;
  std::vector<std::array<int64_t, 10>> faces;
  std::map<int64_t, int64_t> v_to_v_map;
  std::vector<Eigen::Vector3d> vertices;
  const auto &F = m_affine_manifold.get_faces();

  Eigen::MatrixXd edges;
  igl::edges(F, edges);

  Eigen::MatrixXd edges_uv;
  igl::edges(m_affine_manifold.get_F_uv(), edges_uv);

  // checks
  std::cout << "#3d face edges: " << edges.rows() << std::endl;
  std::cout << "#uv face edges: " << edges_uv.rows() << std::endl;
  std::cout << "#edge charts: " << m_affine_manifold.m_edge_charts.size()
            << std::endl;
  for (const auto &e : m_affine_manifold.m_edge_charts) {
    if (e.is_boundary) {
      std::cout << "boundary" << std::endl;
    }
  }

  for (const auto &v : m_affine_manifold.m_vertex_charts) {
    if (v.is_cone) {
      std::cout << "cone" << std::endl;
    }
  }

  // compute corner vertices first
  for (size_t p_idx = 0; p_idx < m_patches.size(); p_idx++) {
    const auto &patch = m_patches[p_idx];
    // idk why but this is  2 0 1 not 0 1 2, maybe because of the half edge data
    // structure
    std::array<int64_t, 3> Fv = {{F(p_idx, 2), F(p_idx, 0), F(p_idx, 1)}};
    for (int i = 0; i < 3; ++i) {
      if (v_to_v_map.find(Fv[i]) != v_to_v_map.end()) {
        // vertex already computed
        continue;
      } else {
        auto z = patch.CT_eval(CT_nodes[i][0], CT_nodes[i][1]);
        v_to_v_map[Fv[i]] = vertices.size();
        vertices.push_back(z);
      }
    }
  }

  assert(size_t(F.rows()) == m_patches.size());

  for (size_t p_idx = 0; p_idx < m_patches.size(); p_idx++) {
    std::array<int64_t, 19> l_vids = {-1};
    // idk why but this is  2 0 1 not 0 1 2, maybe because of the half edge data
    // structure
    std::array<int64_t, 3> Fv = {{F(p_idx, 2), F(p_idx, 0), F(p_idx, 1)}};
    const auto &patch = m_patches[p_idx];

    // node 0 - 2
    for (int i = 0; i < 3; ++i) {
      l_vids[i] = v_to_v_map[Fv[i]];
    }

    // node 3
    auto zz = patch.CT_eval(CT_nodes[3][0], CT_nodes[3][1]);
    l_vids[3] = vertices.size();
    vertices.push_back(zz);

    // node 4 5 6 7 8 9
    for (int i = 0; i < 3; ++i) {
      if (boundary_edge_to_v_map.find(std::make_pair(Fv[(i + 1) % 3], Fv[i])) !=
          boundary_edge_to_v_map.end()) {
        // this edge is processed in some other patch
        const auto &vs =
            boundary_edge_to_v_map[std::make_pair(Fv[(i + 1) % 3], Fv[i])];
        l_vids[4 + i * 2 + 0] = vs[1];
        l_vids[4 + i * 2 + 1] = vs[0];
      } else {
        // eval new vertices
        auto z0 = patch.CT_eval(CT_nodes[4 + i * 2 + 0][0],
                                CT_nodes[4 + i * 2 + 0][1]);
        auto z1 = patch.CT_eval(CT_nodes[4 + i * 2 + 1][0],
                                CT_nodes[4 + i * 2 + 1][1]);
        l_vids[4 + i * 2 + 0] = vertices.size();
        vertices.push_back(z0);
        l_vids[4 + i * 2 + 1] = vertices.size();
        vertices.push_back(z1);

        boundary_edge_to_v_map[std::make_pair(Fv[i], Fv[(i + 1) % 3])] = {
            {l_vids[4 + i * 2 + 0], l_vids[4 + i * 2 + 0 + 1]}};
      }
    }

    // node 10 - 18
    for (int i = 10; i < 19; ++i) {
      auto z = patch.CT_eval(CT_nodes[i][0], CT_nodes[i][1]);
      l_vids[i] = vertices.size();
      vertices.push_back(z);
    }

    /*
    subtri0: 0 1 3 4 5 12 13 11 10 16
    subtri1: 1 2 3 6 7 14 15 13 12 17
    subtri2: 2 0 3 8 9 10 11 15 14 18
    */
    faces.push_back(
        {{l_vids[0] + 1, l_vids[1] + 1, l_vids[3] + 1, l_vids[4] + 1,
          l_vids[5] + 1, l_vids[12] + 1, l_vids[13] + 1, l_vids[11] + 1,
          l_vids[10] + 1, l_vids[16] + 1}});
    faces.push_back(
        {{l_vids[1] + 1, l_vids[2] + 1, l_vids[3] + 1, l_vids[6] + 1,
          l_vids[7] + 1, l_vids[14] + 1, l_vids[15] + 1, l_vids[13] + 1,
          l_vids[12] + 1, l_vids[17] + 1}});
    faces.push_back(
        {{l_vids[2] + 1, l_vids[0] + 1, l_vids[3] + 1, l_vids[8] + 1,
          l_vids[9] + 1, l_vids[10] + 1, l_vids[11] + 1, l_vids[15] + 1,
          l_vids[14] + 1, l_vids[18] + 1}});

    // // debug code
    // for (size_t i = 0; i < l_vids.size(); ++i) {
    //   std::cout << l_vids[i] << ": " << CT_nodes[i] << std::endl;
    // }
    // std::cout << std::endl;
    // if (p_idx == 2)
    //   break;
  }

  file << "$Nodes\n";

  const size_t node_size = vertices.size();
  file << "1 " << node_size << " 1 " << node_size << "\n";
  file << "2 1 0 " << node_size << "\n";

  for (size_t i = 1; i <= node_size; ++i) {
    file << i << "\n";
  }

  for (size_t i = 0; i < node_size; ++i) {
    file << vertices[i][0] << " " << vertices[i][1] << " " << vertices[i][2]
         << "\n";
  }

  file << "$EndNodes\n";

  // write elements
  // assert(m_patches.size() * 3 == faces.size());
  const size_t element_size = faces.size();

  file << "$Elements\n";
  file << "1 " << element_size << " 1 " << element_size << "\n";
  file << "2 1 21 " << element_size << "\n";
  for (size_t i = 0; i < element_size; ++i) {
    file << i + 1 << " ";
    for (int j = 0; j < 10; ++j) {
      file << faces[i][j] << " ";
    }
    file << "\n";
  }

  file << "$EndElements\n";

  // mark cones
  // const auto &cone_indices = m_affine_manifold.generate_cones();

  // file << "$NodeData\n";
  // file << "1\n";                       // num string tags
  // file << "\"Cone\"\n";                // string tag
  // file << "1\n";                       // num real tags
  // file << "0.0\n";                     // time step starts
  // file << "3\n";                       // three integer tags
  // file << "0\n";                       // time step
  // file << "1\n";                       // num field
  // file << cone_indices.size() << "\n"; // num associated nodal values
  // for (const auto &idx : cone_indices) {
  //   file << v_to_v_map[idx] + 1 << " 1.0\n";
  // }
  // file << "$EndNodeData\n";

  std::ofstream v_map_file(filename + "_input_v_to_output_v_map.txt");
  for (const auto &pair : v_to_v_map) {
    v_map_file << pair.first << " " << pair.second << std::endl;
  }
}

void CloughTocherSurface::
    write_cubic_surface_to_msh_with_conn_from_lagrange_nodes(
        std::string filename) {
  std::ofstream file(filename + ".msh");

  /*
    $MeshFormat
      4.1 0 8     MSH4.1, ASCII
      $EndMeshFormat
    */

  file << "$MeshFormat\n"
       << "4.1 0 8\n"
       << "$EndMeshFormat\n";

  /*
  subtri0: 0 1 18 3 4 14 15 13 12 9
  b0 b1 bc b01 b10 b1c bc1 bc0 b0c b01^c
  subtri1: 1 2 18 5 6 16 17 15 14 10
  b1 b2 bc b12 b21 b2c bc2 bc1 b1c b12^c
  subtri2: 2 0 18 7 8 12 13 17 16 11
  b2 b0 bc b20 b02 b0c bc0 bc2 b2c b20^c
  */

  // m_affine_manifold.generate_lagrange_nodes();

  const auto &lagrange_nodes = m_affine_manifold.m_lagrange_nodes;
  // evaluate vertices
  std::vector<Eigen::Vector3d> vertices;
  for (size_t i = 0; i < lagrange_nodes.size(); ++i) {
    const auto patch_idx = lagrange_nodes[i].first;
    const auto bc = lagrange_nodes[i].second;
    auto z = m_patches[patch_idx].CT_eval(bc[0], bc[1]);
    vertices.push_back(z);
  }

  m_lagrange_node_values = vertices;

  // build faces
  std::vector<std::array<int64_t, 10>> faces;
  for (const auto &f_chart : m_affine_manifold.m_face_charts) {
    const auto &l_nodes = f_chart.lagrange_nodes;
    faces.push_back(
        {{l_nodes[0], l_nodes[1], l_nodes[18], l_nodes[3], l_nodes[4],
          l_nodes[14], l_nodes[15], l_nodes[13], l_nodes[12], l_nodes[9]}});
    faces.push_back(
        {{l_nodes[1], l_nodes[2], l_nodes[18], l_nodes[5], l_nodes[6],
          l_nodes[16], l_nodes[17], l_nodes[15], l_nodes[14], l_nodes[10]}});
    faces.push_back(
        {{l_nodes[2], l_nodes[0], l_nodes[18], l_nodes[7], l_nodes[8],
          l_nodes[12], l_nodes[13], l_nodes[17], l_nodes[16], l_nodes[11]}});
  }

  // compute bezier control points
  m_bezier_control_points.resize(m_lagrange_node_values.size());

  Eigen::Matrix<double, 10, 10> p3_lag2bezier_matrix = p3_lag2bezier_m();

  for (const auto &f_chart : m_affine_manifold.m_face_charts) {
    const auto &l_nodes = f_chart.lagrange_nodes;

    std::array<std::array<int64_t, 10>, 3> indices_sub = {
        {{{l_nodes[0], l_nodes[1], l_nodes[18], l_nodes[3], l_nodes[4],
           l_nodes[14], l_nodes[15], l_nodes[13], l_nodes[12], l_nodes[9]}},
         {{l_nodes[1], l_nodes[2], l_nodes[18], l_nodes[5], l_nodes[6],
           l_nodes[16], l_nodes[17], l_nodes[15], l_nodes[14], l_nodes[10]}},
         {{l_nodes[2], l_nodes[0], l_nodes[18], l_nodes[7], l_nodes[8],
           l_nodes[12], l_nodes[13], l_nodes[17], l_nodes[16], l_nodes[11]}}}};

    for (int i = 0; i < 3; ++i) {
      // subtri i
      Eigen::Matrix<double, 10, 3> lag_values_sub, bezier_points_sub;

      for (int k = 0; k < 10; ++k) {
        lag_values_sub.row(k) = m_lagrange_node_values[indices_sub[i][k]];
      }

      // convert local 10 lag to bezier
      bezier_points_sub = p3_lag2bezier_matrix * lag_values_sub;

      for (int k = 0; k < 10; ++k) {
        m_bezier_control_points[indices_sub[i][k]] = bezier_points_sub.row(k);
      }
    }
  }

  // debug use, out put bezier net
  vertices = m_bezier_control_points;

  file << "$Nodes\n";

  const size_t node_size = vertices.size();
  file << "1 " << node_size << " 1 " << node_size << "\n";
  file << "2 1 0 " << node_size << "\n";

  for (size_t i = 1; i <= node_size; ++i) {
    file << i << "\n";
  }

  for (size_t i = 0; i < node_size; ++i) {
    file << std::setprecision(16) << vertices[i][0] << " " << vertices[i][1]
         << " " << vertices[i][2] << "\n";
  }

  file << "$EndNodes\n";

  // write elements
  // assert(m_patches.size() * 3 == faces.size());
  const size_t element_size = faces.size();

  file << "$Elements\n";
  file << "1 " << element_size << " 1 " << element_size << "\n";
  file << "2 1 21 " << element_size << "\n";
  for (size_t i = 0; i < element_size; ++i) {
    file << i + 1 << " ";
    for (int j = 0; j < 10; ++j) {
      file << faces[i][j] + 1 << " ";
    }
    file << "\n";
  }

  file << "$EndElements\n";

  std::ofstream v_map_file(filename + "_input_v_to_output_v_map.txt");
  for (const auto &pair : m_affine_manifold.v_to_lagrange_node_map) {
    v_map_file << pair.first << " " << pair.second << std::endl;
  }
}

Eigen::Vector3d bc_to_cart(const Eigen::Vector3i &f, const Eigen::MatrixXd &V,
                           const PlanarPoint &bc) {
  const double &u = bc[0];
  const double &v = bc[1];
  const double w = 1.0 - u - v;

  return u * V.row(f[0]) + v * V.row(f[1]) + w * V.row(f[2]);
}

void CloughTocherSurface::write_connected_lagrange_nodes(std::string filename,
                                                         Eigen::MatrixXd &V) {
  const auto &lagrange_nodes = m_affine_manifold.m_lagrange_nodes;
  const auto &F = m_affine_manifold.get_faces();
  // evaluate vertices
  std::vector<Eigen::Vector3d> vertices;
  for (size_t i = 0; i < lagrange_nodes.size(); ++i) {
    const auto &patch_idx = lagrange_nodes[i].first;
    const auto &bc = lagrange_nodes[i].second;

    const Eigen::Vector3i &f = F.row(patch_idx);
    vertices.push_back(bc_to_cart(f, V, bc));
  }

  // create faces
  std::vector<Eigen::Vector3i> faces;
  for (const auto &f_chart : m_affine_manifold.m_face_charts) {
    const auto &l_nodes = f_chart.lagrange_nodes;
    // 01c
    faces.emplace_back(l_nodes[0], l_nodes[3], l_nodes[12]);
    faces.emplace_back(l_nodes[12], l_nodes[3], l_nodes[9]);
    faces.emplace_back(l_nodes[3], l_nodes[4], l_nodes[9]);
    faces.emplace_back(l_nodes[9], l_nodes[4], l_nodes[14]);
    faces.emplace_back(l_nodes[4], l_nodes[1], l_nodes[14]);
    faces.emplace_back(l_nodes[12], l_nodes[9], l_nodes[13]);
    faces.emplace_back(l_nodes[13], l_nodes[9], l_nodes[15]);
    faces.emplace_back(l_nodes[9], l_nodes[14], l_nodes[15]);
    faces.emplace_back(l_nodes[13], l_nodes[15], l_nodes[18]);

    // 12c
    faces.emplace_back(l_nodes[1], l_nodes[5], l_nodes[14]);
    faces.emplace_back(l_nodes[14], l_nodes[5], l_nodes[10]);
    faces.emplace_back(l_nodes[5], l_nodes[6], l_nodes[10]);
    faces.emplace_back(l_nodes[10], l_nodes[6], l_nodes[16]);
    faces.emplace_back(l_nodes[6], l_nodes[2], l_nodes[16]);
    faces.emplace_back(l_nodes[14], l_nodes[10], l_nodes[15]);
    faces.emplace_back(l_nodes[15], l_nodes[10], l_nodes[17]);
    faces.emplace_back(l_nodes[10], l_nodes[16], l_nodes[17]);
    faces.emplace_back(l_nodes[15], l_nodes[17], l_nodes[18]);

    // 20c
    faces.emplace_back(l_nodes[2], l_nodes[7], l_nodes[16]);
    faces.emplace_back(l_nodes[16], l_nodes[7], l_nodes[11]);
    faces.emplace_back(l_nodes[7], l_nodes[8], l_nodes[11]);
    faces.emplace_back(l_nodes[11], l_nodes[8], l_nodes[12]);
    faces.emplace_back(l_nodes[8], l_nodes[0], l_nodes[12]);
    faces.emplace_back(l_nodes[16], l_nodes[11], l_nodes[17]);
    faces.emplace_back(l_nodes[17], l_nodes[11], l_nodes[13]);
    faces.emplace_back(l_nodes[11], l_nodes[12], l_nodes[13]);
    faces.emplace_back(l_nodes[17], l_nodes[13], l_nodes[18]);
  }

  std::ofstream file(filename + ".obj");
  for (size_t i = 0; i < vertices.size(); ++i) {
    file << "v " << vertices[i][0] << " " << vertices[i][1] << " "
         << vertices[i][2] << std::endl;
  }

  for (size_t i = 0; i < faces.size(); ++i) {
    file << "f " << faces[i][0] + 1 << " " << faces[i][1] + 1 << " "
         << faces[i][2] + 1 << std::endl;
  }
  file.close();
}

void CloughTocherSurface::write_connected_lagrange_nodes_values(
    std::string filename) {
  const auto &lagrange_nodes = m_affine_manifold.m_lagrange_nodes;
  const auto &F = m_affine_manifold.get_faces();
  // evaluate vertices
  std::vector<Eigen::Vector3d> vertices;
  for (size_t i = 0; i < lagrange_nodes.size(); ++i) {
    const auto &patch_idx = lagrange_nodes[i].first;
    const auto &bc = lagrange_nodes[i].second;
    vertices.push_back(m_patches[patch_idx].CT_eval(bc[0], bc[1]));
  }

  std::vector<int64_t> v_around_cone;
  std::vector<int64_t> f_around_cone;

  // create faces
  std::vector<Eigen::Vector3i> faces;
  for (const auto &f_chart : m_affine_manifold.m_face_charts) {
    const auto &l_nodes = f_chart.lagrange_nodes;

    if (f_chart.is_cone_adjacent) {
      if (m_affine_manifold.m_vertex_charts[F.row(f_chart.face_index)[0]]
              .is_cone) {
        for (int i = 0; i < 19; ++i) {
          if (i != 1 && i != 2 && i != 5 && i != 6 && i != 10)
            v_around_cone.push_back(l_nodes[i]);
        }
      }
      if (m_affine_manifold.m_vertex_charts[F.row(f_chart.face_index)[1]]
              .is_cone) {
        for (int i = 0; i < 19; ++i) {
          if (i != 0 && i != 2 && i != 7 && i != 8 && i != 11)
            v_around_cone.push_back(l_nodes[i]);
        }
      }
      if (m_affine_manifold.m_vertex_charts[F.row(f_chart.face_index)[2]]
              .is_cone) {
        for (int i = 0; i < 19; ++i) {
          if (i != 0 && i != 1 && i != 3 && i != 4 && i != 9)
            v_around_cone.push_back(l_nodes[i]);
        }
      }

      for (int i = 0; i < 27; ++i) {
        f_around_cone.push_back(faces.size() + i);
      }
    }

    // 01c
    faces.emplace_back(l_nodes[0], l_nodes[3], l_nodes[12]);
    faces.emplace_back(l_nodes[12], l_nodes[3], l_nodes[9]);
    faces.emplace_back(l_nodes[3], l_nodes[4], l_nodes[9]);
    faces.emplace_back(l_nodes[9], l_nodes[4], l_nodes[14]);
    faces.emplace_back(l_nodes[4], l_nodes[1], l_nodes[14]);
    faces.emplace_back(l_nodes[12], l_nodes[9], l_nodes[13]);
    faces.emplace_back(l_nodes[13], l_nodes[9], l_nodes[15]);
    faces.emplace_back(l_nodes[9], l_nodes[14], l_nodes[15]);
    faces.emplace_back(l_nodes[13], l_nodes[15], l_nodes[18]);

    // 12c
    faces.emplace_back(l_nodes[1], l_nodes[5], l_nodes[14]);
    faces.emplace_back(l_nodes[14], l_nodes[5], l_nodes[10]);
    faces.emplace_back(l_nodes[5], l_nodes[6], l_nodes[10]);
    faces.emplace_back(l_nodes[10], l_nodes[6], l_nodes[16]);
    faces.emplace_back(l_nodes[6], l_nodes[2], l_nodes[16]);
    faces.emplace_back(l_nodes[14], l_nodes[10], l_nodes[15]);
    faces.emplace_back(l_nodes[15], l_nodes[10], l_nodes[17]);
    faces.emplace_back(l_nodes[10], l_nodes[16], l_nodes[17]);
    faces.emplace_back(l_nodes[15], l_nodes[17], l_nodes[18]);

    // 20c
    faces.emplace_back(l_nodes[2], l_nodes[7], l_nodes[16]);
    faces.emplace_back(l_nodes[16], l_nodes[7], l_nodes[11]);
    faces.emplace_back(l_nodes[7], l_nodes[8], l_nodes[11]);
    faces.emplace_back(l_nodes[11], l_nodes[8], l_nodes[12]);
    faces.emplace_back(l_nodes[8], l_nodes[0], l_nodes[12]);
    faces.emplace_back(l_nodes[16], l_nodes[11], l_nodes[17]);
    faces.emplace_back(l_nodes[17], l_nodes[11], l_nodes[13]);
    faces.emplace_back(l_nodes[11], l_nodes[12], l_nodes[13]);
    faces.emplace_back(l_nodes[17], l_nodes[13], l_nodes[18]);
  }

  std::ofstream file_cone_adj_v(filename + "_cone_area_vertices.txt");
  for (size_t i = 0; i < v_around_cone.size(); ++i) {
    file_cone_adj_v << v_around_cone[i] << std::endl;
  }
  file_cone_adj_v.close();

  std::ofstream file_cone_adj_f(filename + "_cone_area_faces.txt");
  for (size_t i = 0; i < f_around_cone.size(); ++i) {
    file_cone_adj_f << f_around_cone[i] << std::endl;
  }
  file_cone_adj_f.close();

  std::ofstream file(filename + ".obj");
  for (size_t i = 0; i < vertices.size(); ++i) {
    file << "v " << vertices[i][0] << " " << vertices[i][1] << " "
         << vertices[i][2] << std::endl;
  }

  for (size_t i = 0; i < faces.size(); ++i) {
    file << "f " << faces[i][0] + 1 << " " << faces[i][1] + 1 << " "
         << faces[i][2] + 1 << std::endl;
  }
  file.close();
}

void CloughTocherSurface::
    write_external_bd_interpolated_function_values_from_lagrange_nodes(
        std::string filename,
        std::vector<Eigen::Matrix<double, 12, 1>> &external_boundary_data) {
  std::ofstream file(filename + ".txt");
  std::ofstream file_uv(filename + "_uvs.txt");

  const auto &lagrange_nodes = m_affine_manifold.m_lagrange_nodes;
  const auto &f_charts = m_affine_manifold.m_face_charts;
  // evaluate function values
  std::vector<double> values;
  std::vector<Eigen::Vector2d> uv_positions;
  for (size_t i = 0; i < lagrange_nodes.size(); ++i) {
    const auto patch_idx = lagrange_nodes[i].first;
    const auto bc = lagrange_nodes[i].second;
    auto z = m_patches[patch_idx].external_boundary_data_eval(
        bc[0], bc[1], external_boundary_data[patch_idx]);
    values.push_back(z);

    const auto &tri_coords = f_charts[patch_idx].face_uv_positions;
    Eigen::Vector2d uv_pos = bc[0] * tri_coords[0] + bc[1] * tri_coords[1] +
                             (1.0 - bc[0] - bc[1]) * tri_coords[2];
    uv_positions.push_back(uv_pos);
  }

  for (const double &v : values) {
    file << std::setprecision(16) << v << std::endl;
  }

  for (const auto &p : uv_positions) {
    file_uv << p[0] << " " << p[1] << std::endl;
  }

  file.close();
  file_uv.close();
}

void CloughTocherSurface::P_G2F(Eigen::SparseMatrix<double> &m) {
  const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto F_cnt = m_affine_manifold.m_face_charts.size();

  m.resize(19 * F_cnt, N_L);

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(F_cnt * 19);

  const auto &face_charts = m_affine_manifold.m_face_charts;
  for (size_t i = 0; i < face_charts.size(); ++i) {
    for (int j = 0; j < 19; ++j) {
      triplets.emplace_back(i * 19 + j, face_charts[i].lagrange_nodes[j], 1);
    }
  }

  m.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::saveMarket(m, "P_G2F_matrix.txt");
}

void CloughTocherSurface::C_L_int(Eigen::Matrix<double, 7, 19> &m) {
  Eigen::Matrix<double, 12, 12> L_L2d_ind = L_L2d_ind_m();
  Eigen::Matrix<double, 7, 12> L_d2L_dep = L_d2L_dep_m();

  Eigen::Matrix<double, 7, 12> neg_L_dot_L = -L_d2L_dep * L_L2d_ind;
  m.block<7, 12>(0, 0) = neg_L_dot_L;
  m.block<7, 7>(0, 12) = Eigen::MatrixXd::Identity(7, 7);
}

void CloughTocherSurface::C_F_int(Eigen::SparseMatrix<double> &m) {
  const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto F_cnt = m_affine_manifold.m_face_charts.size();

  Eigen::SparseMatrix<double> p_g2f;
  P_G2F(p_g2f);
  Eigen::Matrix<double, 7, 19> c_l_int;
  C_L_int(c_l_int);

  Eigen::SparseMatrix<double> C_diag;
  C_diag.resize(7 * F_cnt, 19 * F_cnt);
  C_diag.reserve(Eigen::VectorXi::Constant(19 * F_cnt, 7));

  for (size_t i = 0; i < F_cnt; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 19; ++k) {
        C_diag.insert(i * 7 + j, i * 19 + k) = c_l_int(j, k);
      }
    }
  }
  C_diag.makeCompressed();

  m.resize(7 * F_cnt, N_L);
  m.reserve(Eigen::VectorXi::Constant(N_L, 7));
  // std::cout << 7 * F_cnt << " " << N_L << std::endl;

  m = C_diag * p_g2f;
  m.makeCompressed();
  // std::cout << m.rows() << " " << m.cols() << std::endl;
}

void CloughTocherSurface::P_G2E(Eigen::SparseMatrix<double> &m) {
  const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto E_cnt = m_affine_manifold.m_edge_charts.size();

  m.resize(24 * E_cnt, N_L);

  const auto &e_charts = m_affine_manifold.m_edge_charts;
  const auto &f_charts = m_affine_manifold.m_face_charts;

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(E_cnt * 24);
  for (size_t i = 0; i < e_charts.size(); ++i) {
    // std::cout << "edge " << e_charts[i].left_vertex_index << " "
    //           << e_charts[i].right_vertex_index << ": " << std::endl;
    const auto &top_face_idx = e_charts[i].top_face_index;
    const auto &bottom_face_idx = e_charts[i].bottom_face_index;
    for (int j = 0; j < 12; ++j) {
      triplets.emplace_back(i * 24 + j,
                            f_charts[top_face_idx].lagrange_nodes[j], 1);
      // std::cout << f_charts[top_face_idx].lagrange_nodes[j] << " ";
    }
    // std::cout << " / ";
    if (bottom_face_idx > -1) {
      for (int j = 0; j < 12; ++j) {
        triplets.emplace_back(i * 24 + 12 + j,
                              f_charts[bottom_face_idx].lagrange_nodes[j], 1);
        // std::cout << f_charts[bottom_face_idx].lagrange_nodes[j] << " ";
      }
    }
    // std::cout << std::endl;
  }

  m.setFromTriplets(triplets.begin(), triplets.end());
}

std::array<int, 4> P_dE_helper(int a, int b) {
  int hash = a * 10 + b;

  // get
  // d01 d02 d10 d12
  // from
  // p0 p1 p2 d01 d10 d12 d21 d20 d02 h01 h12 h20
  // 0  1  2  3   4   5   6   7   8   9   10  11

  switch (hash) {
  case 1:
    // 01
    return {{3, 8, 4, 5}};
  case 12:
    // 12
    return {{5, 4, 6, 7}};
  case 20:
    // 20
    return {{7, 6, 8, 3}};
  case 10:
    // 10
    return {{4, 5, 3, 8}};
  case 21:
    // 21
    return {{6, 7, 5, 4}};
  case 2:
    // 02
    return {{8, 3, 7, 6}};
  default:
    break;
  }

  return {{-1, -1, -1, -1}};
}

Eigen::Matrix2d inverse_2by2(const Eigen::Matrix2d &m) {
  const double &a = m(0, 0);
  const double &b = m(0, 1);
  const double &c = m(1, 0);
  const double &d = m(1, 1);
  double det = (a * d - b * c);
  Eigen::Matrix2d r;
  r << d, -b, -c, a;
  r /= det;
  return r;
}

// abc and cbd
// return cos(theta)
double compute_dihedral_angle(const Eigen::Vector3d &a,
                              const Eigen::Vector3d &b,
                              const Eigen::Vector3d &c,
                              const Eigen::Vector3d &d) {
  Eigen::Vector3d t1_normal = (b - a).cross(c - b);
  Eigen::Vector3d t2_normal = (b - c).cross(d - b);

  return t1_normal.dot(t2_normal) / (t1_normal.norm() * t2_normal.norm());
}

void CloughTocherSurface::C_E_end(Eigen::SparseMatrix<double> &m,
                                  Eigen::SparseMatrix<double> &m_elim) {
  const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto E_cnt = m_affine_manifold.m_edge_charts.size();

  m.resize(2 * E_cnt, N_L);

  const auto &e_charts = m_affine_manifold.m_edge_charts;
  // const auto &f_charts = m_affine_manifold.m_face_charts;
  const auto &v_charts = m_affine_manifold.m_vertex_charts;
  const auto &Fv = m_affine_manifold.get_faces();
  // const auto &Fv_uv = m_affine_manifold.get_F_uv();
  // const auto &uvs = m_affine_manifold.get_global_uv();

  const Eigen::Matrix<double, 12, 12> L_L2d_ind = L_L2d_ind_m();
  Eigen::Matrix<double, 24, 24> diag_L_L2d_ind;
  diag_L_L2d_ind.setZero();
  diag_L_L2d_ind.block<12, 12>(0, 0) = L_L2d_ind;
  diag_L_L2d_ind.block<12, 12>(12, 12) = L_L2d_ind;

  Eigen::SparseMatrix<double> C_E_L;
  C_E_L.resize(2 * E_cnt, 24 * E_cnt);
  C_E_L.reserve(Eigen::VectorXi::Constant(24 * E_cnt, 2));

  // std::vector<int> visited(v_charts.size(), 0);
  std::vector<int> skip(2 * E_cnt, 0);
  int64_t skip_cnt = 0;

  std::vector<int64_t> row_vertex_id_map;

  for (size_t eid = 0; eid < e_charts.size(); ++eid) {
    const auto &e = e_charts[eid];
    if (e.is_boundary) {
      // skip boundary edges
      continue;
    }

    const auto &v0_chart = v_charts[e.left_vertex_index];
    if (!v0_chart.is_cone) {
      // if other vertex is the first or second vertex in one ring
      if (e.right_vertex_index == v0_chart.vertex_one_ring[0] ||
          e.right_vertex_index == v0_chart.vertex_one_ring[1]) {
        skip[2 * eid + 0] = 1;
        skip_cnt++;
      }
    }

    const auto &v1_chart = v_charts[e.right_vertex_index];
    if (!v1_chart.is_cone) {
      if (e.left_vertex_index == v1_chart.vertex_one_ring[0] ||
          e.left_vertex_index == v1_chart.vertex_one_ring[1]) {
        skip[2 * eid + 1] = 1;
        skip_cnt++;
      }
    }

    row_vertex_id_map.push_back(e.left_vertex_index);
    row_vertex_id_map.push_back(e.right_vertex_index);

    // T and T'
    const auto &T = Fv.row(e.top_face_index);
    const auto &T_prime = Fv.row(e.bottom_face_index);

    // v_pos using edge charts
    const auto &v0_pos = e.left_vertex_uv_position;
    const auto &v1_pos = e.right_vertex_uv_position;
    const auto &v2_pos = e.top_vertex_uv_position;

    const auto &v0_pos_prime = e.right_vertex_uv_position;
    const auto &v1_pos_prime = e.left_vertex_uv_position;
    const auto &v2_pos_prime = e.bottom_vertex_uv_position;

    // // v pos using vertex charts, should be equivalant to using e chart cuz
    // // in e chart two triangles are scaled with the same factor
    // // v0 is at origin in v chart
    // // find the v1 v2 v2' in the v chart
    // const auto &v0_idx = e.left_vertex_index;
    // const auto &v1_idx = e.right_vertex_index;
    // const auto &v2_idx = e.top_vertex_index;
    // const auto &v2_prime_idx = e.bottom_vertex_index;
    // const auto &v0_chart = v_charts[v0_idx];
    // const auto &v1_chart = v_charts[v1_idx];

    // PlanarPoint v0_pos, v1_pos, v2_pos, v0_pos_prime, v1_pos_prime,
    //     v2_pos_prime;

    // // use a non-cone vertex to compute, otherwise it won't close up
    // if (!v0_chart.is_cone) {
    //   int v1_lid_in_v_chart = -1, v2_lid_in_v_chart = -1,
    //       v2_prime_lid_in_v_chart = -1;

    //   for (size_t i = 0; i < v0_chart.vertex_one_ring.size(); ++i) {
    //     if (v0_chart.vertex_one_ring[i] == v1_idx) {
    //       v1_lid_in_v_chart = i;
    //     }
    //     if (v0_chart.vertex_one_ring[i] == v2_idx) {
    //       v2_lid_in_v_chart = i;
    //     }
    //     if (v0_chart.vertex_one_ring[i] == v2_prime_idx) {
    //       v2_prime_lid_in_v_chart = i;
    //     }
    //   }

    //   assert(v1_lid_in_v_chart != -1);
    //   assert(v2_lid_in_v_chart != -1);
    //   assert(v2_prime_lid_in_v_chart != -1);

    //   v0_pos = PlanarPoint(0., 0.);
    //   v1_pos = v0_chart.one_ring_uv_positions.row(v1_lid_in_v_chart);
    //   v2_pos = v0_chart.one_ring_uv_positions.row(v2_lid_in_v_chart);

    //   v0_pos_prime = v1_pos;
    //   v1_pos_prime = v0_pos;
    //   v2_pos_prime =
    //       v0_chart.one_ring_uv_positions.row(v2_prime_lid_in_v_chart);
    // } else {
    //   // use v2, if both are cones it's okay because we overwrite cde with
    //   // 1's
    //   int v0_lid_in_v_chart = -1, v2_lid_in_v_chart = -1,
    //       v2_prime_lid_in_v_chart = -1;

    //   for (size_t i = 0; i < v1_chart.vertex_one_ring.size(); ++i) {
    //     if (v1_chart.vertex_one_ring[i] == v0_idx) {
    //       v0_lid_in_v_chart = i;
    //     }
    //     if (v1_chart.vertex_one_ring[i] == v2_idx) {
    //       v2_lid_in_v_chart = i;
    //     }
    //     if (v1_chart.vertex_one_ring[i] == v2_prime_idx) {
    //       v2_prime_lid_in_v_chart = i;
    //     }
    //   }

    //   assert(v0_lid_in_v_chart != -1);
    //   assert(v2_lid_in_v_chart != -1);
    //   assert(v2_prime_lid_in_v_chart != -1);

    //   v0_pos = v1_chart.one_ring_uv_positions.row(v0_lid_in_v_chart);
    //   v1_pos = PlanarPoint(0., 0.);
    //   v2_pos = v1_chart.one_ring_uv_positions.row(v2_lid_in_v_chart);

    //   v0_pos_prime = v1_pos;
    //   v1_pos_prime = v0_pos;
    //   v2_pos_prime =
    //       v1_chart.one_ring_uv_positions.row(v2_prime_lid_in_v_chart);
    // }

    // u_ij

    const auto u_01 = v1_pos - v0_pos;
    const auto u_02 = v2_pos - v0_pos;
    const auto u_12 = v2_pos - v1_pos;
    const auto u_10 = v0_pos - v1_pos;

    // std::cout << "u_01: " << u_01[0] << " " << u_01[1] << std::endl;
    // std::cout << "u_02: " << u_02[0] << " " << u_02[1] << std::endl;
    // std::cout << "u_12: " << u_12[0] << " " << u_12[1] << std::endl;
    // std::cout << "u_10: " << u_10[0] << " " << u_10[1] << std::endl;

    // u_ij_prime
    const auto u_01_prime = v1_pos_prime - v0_pos_prime;
    const auto u_02_prime = v2_pos_prime - v0_pos_prime;
    const auto u_12_prime = v2_pos_prime - v1_pos_prime;
    const auto u_10_prime = v0_pos_prime - v1_pos_prime;

    // std::cout << "u_01': " << u_01_prime[0] << " " << u_01_prime[1]
    //           << std::endl;
    // std::cout << "u_02': " << u_02_prime[0] << " " << u_02_prime[1]
    //           << std::endl;
    // std::cout << "u_12': " << u_12_prime[0] << " " << u_12_prime[1]
    //           << std::endl;
    // std::cout << "u_10': " << u_10_prime[0] << " " << u_10_prime[1]
    //           << std::endl;

    // D0 D1
    Eigen::Matrix<double, 2, 2> D0;
    D0 << u_01[0], u_01[1], u_02[0], u_02[1];
    Eigen::Matrix<double, 2, 2> D1;
    D1 << u_10[0], u_10[1], u_12[0], u_12[1];

    // D0_prime D1_prime
    Eigen::Matrix<double, 2, 2> D0_prime;
    D0_prime << u_01_prime[0], u_01_prime[1], u_02_prime[0], u_02_prime[1];
    Eigen::Matrix<double, 2, 2> D1_prime;
    D1_prime << u_10_prime[0], u_10_prime[1], u_12_prime[0], u_12_prime[1];

    // u_01_prep u_01_prep_prime
    Eigen::Vector2d u_01_prep(-u_01[1], u_01[0]);
    Eigen::Vector2d u_01_prep_prime(-u_01_prime[1], u_01_prime[0]);

    // g0 g1
    Eigen::Matrix<double, 1, 2> g0 = u_01_prep.transpose() * inverse_2by2(D0);
    Eigen::Matrix<double, 1, 2> g1 = u_01_prep.transpose() * inverse_2by2(D1);

    // g0_prime g1_prime
    Eigen::Matrix<double, 1, 2> g0_prime =
        u_01_prep.transpose() * inverse_2by2(D0_prime);
    Eigen::Matrix<double, 1, 2> g1_prime =
        u_01_prep.transpose() * inverse_2by2(D1_prime);

    // C_dE(e)
    Eigen::Matrix<double, 2, 8> C_dE;
    C_dE.row(0) << g0(0, 0), g0(0, 1), -g1_prime(0, 0), -g1_prime(0, 1), 0, 0,
        0,
        0; // v0
    C_dE.row(1) << 0, 0, 0, 0, g1(0, 0), g1(0, 1), -g0_prime(0, 0),
        -g0_prime(0, 1); // v1

    // check cones and modify C_dE(e)
    if (v_charts[e.left_vertex_index].is_cone) {
      // v0 is cone
      C_dE.row(0) << 1, 0, 0, 0, 0, 0, 0, 0;
    }
    if (v_charts[e.right_vertex_index].is_cone) {
      // v1 is cone
      C_dE.row(1) << 0, 0, 0, 0, 1, 0, 0, 0;
    }

    // P_dE reindexing global dof to local index
    Eigen::Matrix<double, 8, 24> P_dE;
    P_dE.setZero();

    // v0 v1 local id in T
    int64_t lid_0, lid_1 = -1;
    for (int i = 0; i < 3; ++i) {
      if (T[i] == e.left_vertex_index) {
        lid_0 = i;
      }
      if (T[i] == e.right_vertex_index) {
        lid_1 = i;
      }
    }
    assert(lid_0 > -1 && lid_1 > -1);

    // std::cout << "T local id for v0 v1: " << lid_0 << " " << lid_1 <<
    // std::endl;

    // v0_prime v1_prime local id in T'
    int64_t lid_0_prime, lid_1_prime = -1;
    for (int i = 0; i < 3; ++i) {
      if (T_prime[i] == e.right_vertex_index) {
        lid_0_prime = i;
      }
      if (T_prime[i] == e.left_vertex_index) {
        lid_1_prime = i;
      }
    }
    assert(lid_0_prime > -1 && lid_1_prime > -1);

    // std::cout << "T' local id for v0 v1: " << lid_0_prime << " " <<
    // lid_1_prime
    //           << std::endl;

    // T and T' reindex
    auto T_P_dE_indices = P_dE_helper(lid_0, lid_1);
    auto T_prime_P_dE_indices = P_dE_helper(lid_0_prime, lid_1_prime);

    // assemble P_dE respect to [qL_int^T qL_int'^T]
    P_dE(0, T_P_dE_indices[0]) = 1;            // d01
    P_dE(1, T_P_dE_indices[1]) = 1;            // d02
    P_dE(2, T_prime_P_dE_indices[2] + 12) = 1; // d10'
    P_dE(3, T_prime_P_dE_indices[3] + 12) = 1; // d12'
    P_dE(4, T_P_dE_indices[2]) = 1;            // d10
    P_dE(5, T_P_dE_indices[3]) = 1;            // d12
    P_dE(6, T_prime_P_dE_indices[0] + 12) = 1; // d01'
    P_dE(7, T_prime_P_dE_indices[1] + 12) = 1; // d02'

    // C_E_L(e)
    Eigen::Matrix<double, 2, 24> C_E_L_e = C_dE * P_dE * diag_L_L2d_ind;

    // set C_E_L
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 24; ++j) {
        C_E_L.insert(eid * 2 + i, eid * 24 + j) = C_E_L_e(i, j);
      }
    }
  }

  C_E_L.makeCompressed();

  Eigen::SparseMatrix<double> p_g2e;
  P_G2E(p_g2e);

  auto timer = igl::Timer();
  timer.start();
  m = C_E_L * p_g2e;
  m.makeCompressed();
  std::cout << "C_E_L * p_g2e time: " << timer.getElapsedTime() << "s"
            << std::endl;

  // compute C_E_L_elim
  Eigen::SparseMatrix<double> C_E_L_elim;
  C_E_L_elim.resize(2 * E_cnt - skip_cnt, 24 * E_cnt);
  C_E_L_elim.reserve(Eigen::VectorXi::Constant(24 * E_cnt, 2));

  std::vector<int> cones;
  m_affine_manifold.compute_cones(cones);
  assert(skip_cnt / 2 == int64_t(v_charts.size() - cones.size()));
  std::cout << "skip cnt: " << skip_cnt
            << " v_cnt - cone_cnt: " << v_charts.size() - cones.size()
            << std::endl;

  int64_t row_id = 0;
  std::vector<int64_t> C_to_C_elim_idx_map(2 * E_cnt, -1);
  for (size_t i = 0; i < skip.size(); ++i) {
    if (skip[i] == 1) {
      continue;
    } else {
      C_to_C_elim_idx_map[i] = row_id;
      row_id++;
    }
  }
  for (int64_t k = 0; k < C_E_L.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(C_E_L, k); it; ++it) {
      if (C_to_C_elim_idx_map[it.row()] != -1) {
        C_E_L_elim.insert(C_to_C_elim_idx_map[it.row()], it.col()) = it.value();
      }
    }
  }
  C_E_L_elim.makeCompressed();

  // get m_elim
  m_elim.resize(2 * E_cnt - skip_cnt, N_L);
  m_elim = C_E_L_elim * p_g2e;

  m_elim.makeCompressed();

  // debug use
  std::ofstream file("endpoint_row_to_vid.txt");
  for (size_t i = 0; i < row_vertex_id_map.size(); ++i) {
    file << row_vertex_id_map[i] << std::endl;
  }
  file.close();
}

std::array<int, 5> P_dM_helper(int a, int b) {
  int hash = a * 10 + b;

  // p0 p1 p01 p10 h01
  // p0 p1 p2 d01 d10 d12 d21 d20 d02 h01 h12 h20
  // 0  1  2  3   4   5   6   7   8   9   10  11

  switch (hash) {
  case 1:
    // 01
    return {{0, 1, 3, 4, 9}};
  case 12:
    // 12
    return {{1, 2, 5, 6, 10}};
  case 20:
    // 20
    return {{2, 0, 7, 8, 11}};
  case 10:
    // 10
    return {{1, 0, 4, 3, 9}};
  case 21:
    // 21
    return {{2, 1, 6, 5, 10}};
  case 2:
    // 02
    return {{0, 2, 8, 7, 11}};
  default:
    break;
  }

  assert(false);

  return {{-1, -1, -1, -1, -1}};
}

void CloughTocherSurface::C_E_mid(Eigen::SparseMatrix<double> &m) {
  const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto E_cnt = m_affine_manifold.m_edge_charts.size();

  m.resize(E_cnt, N_L);

  const auto &e_charts = m_affine_manifold.m_edge_charts;
  // const auto &f_charts = m_affine_manifold.m_face_charts;
  // const auto &v_charts = m_affine_manifold.m_vertex_charts;
  const auto &Fv = m_affine_manifold.get_faces();
  // const auto &Fv_uv = m_affine_manifold.get_F_uv();
  // const auto &uvs = m_affine_manifold.get_global_uv();

  const Eigen::Matrix<double, 12, 12> L_L2d_ind = L_L2d_ind_m();
  Eigen::Matrix<double, 24, 24> diag_L_L2d_ind;
  diag_L_L2d_ind.setZero();
  diag_L_L2d_ind.block<12, 12>(0, 0) = L_L2d_ind;
  diag_L_L2d_ind.block<12, 12>(12, 12) = L_L2d_ind;

  Eigen::Matrix<double, 5, 1> c_h;
  c_h << 0, 0, 0, 0, 1;

  const Eigen::Matrix<double, 5, 1> c_e = c_e_m();

  Eigen::SparseMatrix<double> C_M_L;
  C_M_L.resize(E_cnt, 24 * E_cnt);
  C_M_L.reserve(Eigen::VectorXi::Constant(24 * E_cnt, 1));

  for (size_t eid = 0; eid < e_charts.size(); ++eid) {
    const auto &e = e_charts[eid];
    if (e.is_boundary) {
      // skip boundary edges
      continue;
    }

    // T and T'
    const auto &T = Fv.row(e.top_face_index);
    const auto &T_prime = Fv.row(e.bottom_face_index);

    // v_pos and v_pos'
    const auto &v0_pos = e.left_vertex_uv_position;
    const auto &v1_pos = e.right_vertex_uv_position;
    const auto &v2_pos = e.top_vertex_uv_position;

    const auto &v0_pos_prime = e.right_vertex_uv_position;
    const auto &v1_pos_prime = e.left_vertex_uv_position;
    const auto &v2_pos_prime = e.bottom_vertex_uv_position;

    // u_ij and u_ij'
    const auto u_01 = v1_pos - v0_pos;
    const auto u_02 = v2_pos - v0_pos;
    const auto u_12 = v2_pos - v1_pos;

    const auto u_01_prime = v1_pos_prime - v0_pos_prime;
    const auto u_02_prime = v2_pos_prime - v0_pos_prime;
    const auto u_12_prime = v2_pos_prime - v1_pos_prime;

    // m01 and m01_prime
    const auto m_01 = (u_02 + u_12) / 2.0;
    const auto m_01_prime = (u_02_prime + u_12_prime) / 2.0;

    // u_01_prep u_01_prep_prime
    Eigen::Vector2d u_01_prep(-u_01[1], u_01[0]);
    Eigen::Vector2d u_01_prep_prime(-u_01_prime[1], u_01_prime[0]);

    // g_M and g_M_prime
    Eigen::Matrix<double, 1, 5> g_M;
    g_M =
        (c_h - (m_01.dot(u_01.normalized()) / u_01.norm()) * c_e).transpose() /
        (m_01.dot(u_01_prep.normalized()));
    Eigen::Matrix<double, 1, 5> g_M_prime;
    g_M_prime =
        (c_h -
         (m_01_prime.dot(u_01_prime.normalized()) / u_01_prime.norm()) * c_e)
            .transpose() /
        (m_01_prime.dot(u_01_prep_prime.normalized()));

    // C_dM
    Eigen::Matrix<double, 1, 10> C_dM;
    C_dM.block<1, 5>(0, 0) = g_M;
    C_dM.block<1, 5>(0, 5) = g_M_prime;

    // P_dM
    Eigen::Matrix<double, 10, 24> P_dM;
    P_dM.setZero();

    // v0 v1 local id in T
    int64_t lid_0, lid_1 = -1;
    for (int i = 0; i < 3; ++i) {
      if (T[i] == e.left_vertex_index) {
        lid_0 = i;
      }
      if (T[i] == e.right_vertex_index) {
        lid_1 = i;
      }
    }
    assert(lid_0 > -1 && lid_1 > -1);

    // v0_prime v1_prime local id in T'
    int64_t lid_0_prime, lid_1_prime = -1;
    for (int i = 0; i < 3; ++i) {
      if (T_prime[i] == e.right_vertex_index) {
        lid_0_prime = i;
      }
      if (T_prime[i] == e.left_vertex_index) {
        lid_1_prime = i;
      }
    }
    assert(lid_0_prime > -1 && lid_1_prime > -1);

    // T and T' reindex
    auto T_P_dM_indices = P_dM_helper(lid_0, lid_1);
    auto T_prime_P_dM_indices = P_dM_helper(lid_0_prime, lid_1_prime);

    // assemble P_dM respect to [qL_int  qL_int']
    P_dM(0, T_P_dM_indices[0]) = 1;            // p0
    P_dM(1, T_P_dM_indices[1]) = 1;            // p1
    P_dM(2, T_P_dM_indices[2]) = 1;            // d01
    P_dM(3, T_P_dM_indices[3]) = 1;            // d10
    P_dM(4, T_P_dM_indices[4]) = 1;            // h01
    P_dM(5, T_prime_P_dM_indices[0] + 12) = 1; // p0'
    P_dM(6, T_prime_P_dM_indices[1] + 12) = 1; // p1'
    P_dM(7, T_prime_P_dM_indices[2] + 12) = 1; // d01'
    P_dM(8, T_prime_P_dM_indices[3] + 12) = 1; // d10'
    P_dM(9, T_prime_P_dM_indices[4] + 12) = 1; // h01'

    // C_M_L_e
    Eigen::Matrix<double, 1, 24> C_M_L_e = C_dM * P_dM * diag_L_L2d_ind;

    // set C_M_L
    for (int i = 0; i < 24; ++i) {
      C_M_L.insert(eid, eid * 24 + i) = C_M_L_e(0, i);
    }
  }
  C_M_L.makeCompressed();

  Eigen::SparseMatrix<double> p_g2e;
  P_G2E(p_g2e);

  auto timer = igl::Timer();
  timer.start();
  m = C_M_L * p_g2e;
  m.makeCompressed();
  std::cout << "C_M_L * p_g2e time: " << timer.getElapsedTime() << "s"
            << std::endl;
}

void CloughTocherSurface::diag_P_G2F(Eigen::SparseMatrix<double> &m) {
  const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto F_cnt = m_affine_manifold.m_face_charts.size();

  m.resize(3 * 19 * F_cnt, 3 * N_L);

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(F_cnt * 19 * 3);

  const auto &face_charts = m_affine_manifold.m_face_charts;
  for (size_t i = 0; i < F_cnt; ++i) {
    for (int j = 0; j < 19; ++j) {
      triplets.emplace_back(0 * 19 * F_cnt + i * 19 + j,
                            0 * N_L + face_charts[i].lagrange_nodes[j],
                            1); // block 0
      triplets.emplace_back(1 * 19 * F_cnt + i * 19 + j,
                            1 * N_L + face_charts[i].lagrange_nodes[j],
                            1); // block 1
      triplets.emplace_back(2 * 19 * F_cnt + i * 19 + j,
                            2 * N_L + face_charts[i].lagrange_nodes[j],
                            1); // block 2
    }
  }

  m.setFromTriplets(triplets.begin(), triplets.end());
}

void CloughTocherSurface::P_3D(Eigen::SparseMatrix<double> &m) {
  // const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto F_cnt = m_affine_manifold.m_face_charts.size();

  m.resize(F_cnt * 19 * 3, F_cnt * 19 * 3);

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(F_cnt * 19 * 3);

  for (size_t i = 0; i < F_cnt; ++i) {
    for (int j = 0; j < 19; ++j) {
      triplets.emplace_back(i * 19 * 3 + 0 * 19 + j,
                            0 * F_cnt * 19 + i * 19 + j,
                            1); // x
      triplets.emplace_back(i * 19 * 3 + 1 * 19 + j,
                            1 * F_cnt * 19 + i * 19 + j,
                            1); // y
      triplets.emplace_back(i * 19 * 3 + 2 * 19 + j,
                            2 * F_cnt * 19 + i * 19 + j,
                            1); // z
    }
  }

  m.setFromTriplets(triplets.begin(), triplets.end());
}

std::array<int64_t, 57> P_C_2_helper(const int &lid) {
  // return (row, col) where row is 0 to 56
  std::array<int64_t, 57> row_col;
  std::array<int64_t, 19> row_col_1D;
  switch (lid) {
  case 0:
    row_col_1D = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}};
    break;
  case 1:
    row_col_1D = {
        {1, 2, 0, 5, 6, 7, 8, 3, 4, 10, 11, 9, 14, 15, 16, 17, 12, 13, 18}};
    break;
  case 2:
    row_col_1D = {
        {2, 0, 1, 7, 8, 3, 4, 5, 6, 11, 9, 10, 16, 17, 12, 13, 14, 15, 18}};
    break;
  default:
    assert(false);
    break;
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 19; ++j) {
      row_col[i * 19 + j] = row_col_1D[j] + i * 19;
    }
  }

  return row_col;
}

std::array<int64_t, 36> P_C_2_alt_helper(const int &lid) {
  // return (row, col) where row is 0 to 56
  std::array<int64_t, 36> row_col;
  std::array<int64_t, 12> row_col_1D;
  switch (lid) {
  case 0:
    row_col_1D = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    break;
  case 1:
    row_col_1D = {{1, 2, 0, 5, 6, 7, 8, 3, 4, 10, 11, 9}};
    break;
  case 2:
    row_col_1D = {{2, 0, 1, 7, 8, 3, 4, 5, 6, 11, 9, 10}};
    break;
  default:
    assert(false);
    break;
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 12; ++j) {
      row_col[i * 12 + j] = row_col_1D[j] + i * 12;
    }
  }

  return row_col;
}

void CloughTocherSurface::C_F_cone(Eigen::SparseMatrix<double> &m,
                                   Eigen::MatrixXd &v_normals) {
  // const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto F_cnt = m_affine_manifold.m_face_charts.size();

  // L_L2d
  const Eigen::Matrix<double, 12, 12> L_L2d_ind = L_L2d_ind_m();
  Eigen::Matrix<double, 12, 19> L_L2d;
  L_L2d.block<12, 12>(0, 0) = L_L2d_ind;
  L_L2d.block<12, 7>(0, 12) = Eigen::Matrix<double, 12, 7>::Zero();

  // diag_L_L2d_3
  // Eigen::Matrix<double, 36, 57> diag_L_L2d_3;
  // diag_L_L2d_3.block<12, 19>(0, 0) = L_L2d;
  // diag_L_L2d_3.block<12, 19>(12, 19) = L_L2d;
  // diag_L_L2d_3.block<12, 19>(24, 38) = L_L2d;

  // diag_P_G2F
  Eigen::SparseMatrix<double> diag_p_g2f;
  diag_P_G2F(diag_p_g2f);

  // P_3D
  Eigen::SparseMatrix<double> p_3d;
  P_3D(p_3d);

  // c_t
  const auto &c_t = c_t_m(); // 3 x 3 x Matrix<1, 12>
  // for (int i = 0; i < 3; ++i) {
  //   for (int j = 0; j < 3; ++j) {
  //     std::cout << c_t[i][j] << std::endl;
  //   }
  // }

  const auto &v_charts = m_affine_manifold.m_vertex_charts;
  // const auto &f_charts = m_affine_manifold.m_face_charts;
  const auto &Fv = m_affine_manifold.get_faces();

  // diag_C_cone
  std::vector<Eigen::Triplet<double>> diag_C_cone_triplets;
  // diag_L_L2d
  std::vector<Eigen::Triplet<double>> diag_L_L2d_triplets;
  // P_C_1 N_FC by 3 * N_L
  std::vector<Eigen::Triplet<double>> P_C_1_triplets;
  // P_C_2
  std::vector<Eigen::Triplet<double>> P_C_2_triplets;
  // std::vector<Eigen::Triplet<double>> P_C_2_triplets_alt;

  int64_t N_FC = 0; // number of blocks
  for (size_t vid = 0; vid < v_charts.size(); ++vid) {
    if (!v_charts[vid].is_cone) {
      continue;
    }
    N_FC += v_charts[vid].face_one_ring.size();
  }

  std::vector<int64_t> cone_vids;
  std::vector<int64_t> edge_vids;

  int64_t cone_adj_face_cnt = 0; // face can appear many times in this, plays
                                 // the block id role, == N_FC at last

  for (size_t vid = 0; vid < v_charts.size(); ++vid) {
    const auto &v_chart = v_charts[vid];
    if (!v_chart.is_cone) {
      // skip non cones
      continue;
    }
    cone_vids.push_back(vid);

    const Eigen::Vector3d v_normal =
        v_normals.row(vid); // normal of this cone vertex
    const double nx = v_normal[0];
    const double ny = v_normal[1];
    const double nz = v_normal[2];

    // std::cout << nx << " " << ny << " " << nz << std::endl;
    const auto &v_one_ring_face = v_chart.face_one_ring;

    // C_cone_vid
    Eigen::Matrix<double, 4, 36> C_cone_vid;
    C_cone_vid.block<1, 12>(0, 0) = nx * c_t[0][0];
    C_cone_vid.block<1, 12>(0, 12) = ny * c_t[0][0];
    C_cone_vid.block<1, 12>(0, 24) = nz * c_t[0][0];
    C_cone_vid.block<1, 12>(1, 0) = nx * c_t[0][1];
    C_cone_vid.block<1, 12>(1, 12) = ny * c_t[0][1];
    C_cone_vid.block<1, 12>(1, 24) = nz * c_t[0][1];
    C_cone_vid.block<1, 12>(2, 0) = nx * c_t[2][0];
    C_cone_vid.block<1, 12>(2, 12) = ny * c_t[2][0];
    C_cone_vid.block<1, 12>(2, 24) = nz * c_t[2][0];
    C_cone_vid.block<1, 12>(3, 0) = nx * c_t[2][1];
    C_cone_vid.block<1, 12>(3, 12) = ny * c_t[2][1];
    C_cone_vid.block<1, 12>(3, 24) = nz * c_t[2][1];
    // std::cout << nx * c_t[0][0] << std::endl;
    // std::cout << nx * c_t[2][0] << std::endl;

    // std::cout << C_cone_vid << std::endl << std::endl;

    for (const auto &fid : v_one_ring_face) {
      // std::cout << "------------------------" << std::endl;
      // std::cout << "cone " << vid << " face " << fid << std::endl;
      const auto &T = Fv.row(fid);
      int lid = -1; // local vid of the cone in T
      for (int i = 0; i < 3; ++i) {
        if (int64_t(vid) == T[i]) {
          lid = i;
          break;
        }
      }
      assert(lid > -1);

      edge_vids.push_back(vid);
      edge_vids.push_back(T[(lid + 2) % 3]);

      // std::cout << "T: " << T << std::endl;
      // std::cout << "lid: " << lid << std::endl;

      // add an identity block for P_C_1
      for (int i = 0; i < 57; ++i) { // 57 = 19 * 3 = all x, y, z of a face
        P_C_1_triplets.emplace_back(cone_adj_face_cnt * 57 + i, fid * 57 + i,
                                    1);
      }

      // add a permutation 57 * 57 block for local v index for P_C_2
      const auto &p_c_2_local_row_col = P_C_2_helper(lid);
      for (int i = 0; i < 57; ++i) {
        P_C_2_triplets.emplace_back(
            cone_adj_face_cnt * 57 + i,
            cone_adj_face_cnt * 57 + p_c_2_local_row_col[i], 1);
      }

      // alt
      // const auto &p_c_2_local_row_col_alt = P_C_2_alt_helper(lid);
      // // for (int i = 0; i < 36; ++i) {
      // //   std::cout << p_c_2_local_row_col_alt[i] << " ";
      // // }
      // // std::cout << std::endl;
      // for (int i = 0; i < 36; ++i) {
      //   P_C_2_triplets_alt.emplace_back(
      //       cone_adj_face_cnt * 36 + i,
      //       cone_adj_face_cnt * 36 + p_c_2_local_row_col_alt[i], 1);
      // }

      // diag_L_L2d
      for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 19; ++j) {
          diag_L_L2d_triplets.emplace_back(cone_adj_face_cnt * 36 + 0 * 12 + i,
                                           cone_adj_face_cnt * 57 + 0 * 19 + j,
                                           L_L2d(i, j));
          diag_L_L2d_triplets.emplace_back(cone_adj_face_cnt * 36 + 1 * 12 + i,
                                           cone_adj_face_cnt * 57 + 1 * 19 + j,
                                           L_L2d(i, j));
          diag_L_L2d_triplets.emplace_back(cone_adj_face_cnt * 36 + 2 * 12 + i,
                                           cone_adj_face_cnt * 57 + 2 * 19 + j,
                                           L_L2d(i, j));
        }
      }

      // diag_C_cone
      for (int i = 0; i < 4; ++i) {
        // TODO: drop two rows
        if (i == 2 || i == 3) {
          // drop the third and forth row
          continue;
        }
        for (int j = 0; j < 36; ++j) {
          diag_C_cone_triplets.emplace_back(cone_adj_face_cnt * 2 + i,
                                            cone_adj_face_cnt * 36 + j,
                                            C_cone_vid(i, j));
        }
      }

      cone_adj_face_cnt++;
    }
  }
  assert(cone_adj_face_cnt == N_FC);

  // diag_C_cone
  Eigen::SparseMatrix<double> diag_C_cone;
  diag_C_cone.resize(N_FC * 2, N_FC * 36);
  diag_C_cone.setFromTriplets(diag_C_cone_triplets.begin(),
                              diag_C_cone_triplets.end());

  // diag_L_L2d
  Eigen::SparseMatrix<double> diag_L_L2d;
  diag_L_L2d.resize(N_FC * 36, N_FC * 57);
  diag_L_L2d.setFromTriplets(diag_L_L2d_triplets.begin(),
                             diag_L_L2d_triplets.end());

  // P_C_2
  Eigen::SparseMatrix<double> p_c_2;
  p_c_2.resize(N_FC * 57, N_FC * 57);
  p_c_2.setFromTriplets(P_C_2_triplets.begin(), P_C_2_triplets.end());

  // P_C_2 alt
  // Eigen::SparseMatrix<double> p_c_2_alt;
  // p_c_2_alt.resize(N_FC * 36, N_FC * 36);
  // p_c_2_alt.setFromTriplets(P_C_2_triplets_alt.begin(),
  //                           P_C_2_triplets_alt.end());

  // P_C_1
  Eigen::SparseMatrix<double> p_c_1;
  p_c_1.resize(N_FC * 57, F_cnt * 57);
  p_c_1.setFromTriplets(P_C_1_triplets.begin(), P_C_1_triplets.end());

  m = diag_C_cone * diag_L_L2d * p_c_2 * p_c_1 * p_3d * diag_p_g2f;
  // m = diag_C_cone * p_c_2_alt * diag_L_L2d * p_c_1 * p_3d * diag_p_g2f;
  m.makeCompressed();

  // debug use
  std::ofstream file("cone_vids.txt");
  for (size_t i = 0; i < cone_vids.size(); ++i) {
    file << cone_vids[i] << std::endl;
  }
  file.close();
  std::ofstream file2("cone_edge_vids.txt");
  for (size_t i = 0; i < edge_vids.size(); ++i) {
    file2 << edge_vids[i] << std::endl;
  }
  file2.close();
}

/*
bezier constraints
*/

void assign_spvec_to_spmat_row(Eigen::SparseMatrix<double> &mat,
                               Eigen::SparseVector<double> &vec,
                               const int row) {
  for (Eigen::SparseVector<double>::InnerIterator it(vec); it; ++it) {
    mat.coeffRef(row, it.index()) = it.value();
  }
}

void CloughTocherSurface::Ci_endpoint_ind2dep(
    Eigen::SparseMatrix<double> &m, std::vector<int64_t> &constrained_row_ids) {
  const auto &v_charts = m_affine_manifold.m_vertex_charts;
  // const auto &f_charts = m_affine_manifold.m_face_charts;
  const auto &F = m_affine_manifold.get_faces();

  for (size_t v_id = 0; v_id < v_charts.size(); ++v_id) {
    // // debug use
    // if (v_id > 2) {
    //   break;
    // }

    // // debug use
    // if (v_id == 2) {
    //   std::cout << "in 2" << std::endl;
    // }

    int vid = v_id;

    const auto &v_chart = v_charts[vid];
    const auto &v_one_ring = v_chart.vertex_one_ring;
    const auto &f_one_ring = v_chart.face_one_ring;

    // get vertex uv positions
    std::map<int64_t, Eigen::Vector2d> one_ring_uv_positions_map;
    for (int i = 0; i < v_chart.one_ring_uv_positions.rows(); ++i) {
      one_ring_uv_positions_map[v_one_ring[i]] =
          v_chart.one_ring_uv_positions.row(i);
    }

    std::map<int64_t, bool> processed_id; // vid processed, caution! not node id

    processed_id[vid] = true;

    // pick nodes on the first face to be independent
    // find the two edges
    const auto &first_fid = f_one_ring[0];
    const auto &first_Fv = F.row(first_fid);
    // const auto &first_f_chart = f_charts[first_fid];

    std::vector<int64_t>
        indep_node_ids; // pi pij pik (need to be size 3 after push)
    std::vector<Eigen::Vector2d>
        u_ijik; // uij uik (need to be size 2 after push)

    // find the node id of vid and push it into indep node ids
    for (int i = 0; i < 3; ++i) {
      // find the edges connected to vid
      if (first_Fv[i] != vid) {
        const auto &e_chart = m_affine_manifold.get_edge_chart(first_fid, i);

        if (e_chart.left_vertex_index == vid) {
          indep_node_ids.push_back(e_chart.lagrange_nodes[0]);
        } else {
          indep_node_ids.push_back(e_chart.lagrange_nodes[3]);
        }

        // break after found
        break;
      }
    }

    for (int i = 0; i < 3; ++i) {
      // find the edges connected to vid
      if (first_Fv[i] != vid) {
        const auto &e_chart = m_affine_manifold.get_edge_chart(first_fid, i);

        // get the indep node id on this edge
        if (e_chart.left_vertex_index == vid) {
          indep_node_ids.push_back(e_chart.lagrange_nodes[1]);
          Eigen::Vector2d uij =
              one_ring_uv_positions_map[e_chart.right_vertex_index] -
              Eigen::Vector2d(0, 0);
          // Eigen::Vector2d uij = e_chart.right_global_uv_position -
          //                       e_chart.left_global_uv_position;
          u_ijik.push_back(uij);
          processed_id[e_chart.right_vertex_index] = true;
        } else {
          indep_node_ids.push_back(e_chart.lagrange_nodes[2]);
          Eigen::Vector2d uij =
              one_ring_uv_positions_map[e_chart.left_vertex_index] -
              Eigen::Vector2d(0, 0);
          // Eigen::Vector2d uij = e_chart.left_global_uv_position -
          //                       e_chart.right_global_uv_position;
          u_ijik.push_back(uij);
          processed_id[e_chart.left_vertex_index] = true;
        }

        //   processed_id[first_Fv[i]] = true;
      }
    }

    assert(indep_node_ids.size() == 3);
    assert(u_ijik.size() == 2);

    Eigen::Matrix2d U_ijik;
    U_ijik << u_ijik[0][0], u_ijik[1][0], u_ijik[0][1], u_ijik[1][1];
    Eigen::Matrix2d U_ijik_inv = inverse_2by2(U_ijik);

    // set indep vids in the whole matrix
    if (!v_chart.is_cone) {
      // not cone
      for (int i = 0; i < 3; ++i) {
        m.insert(indep_node_ids[i], indep_node_ids[i]) = 1;
        constrained_row_ids.push_back(indep_node_ids[i]);
      }
      // return;
    } else {
      // cone
      for (int i = 0; i < 3; ++i) {
        m.insert(indep_node_ids[i], indep_node_ids[0]) = 1;
        constrained_row_ids.push_back(indep_node_ids[i]);
      }
    }

    // iterate each face, process other dep vertices
    for (const auto &fid : f_one_ring) {
      for (int i = 0; i < 3; ++i) {
        if (F.row(fid)[i] != vid) {
          const auto &e_chart = m_affine_manifold.get_edge_chart(fid, i);

          // find a unprocessed vertex to process
          if (e_chart.left_vertex_index != vid &&
              processed_id.find(e_chart.left_vertex_index) ==
                  processed_id.end()) {

            // get node id
            auto dep_node_id =
                e_chart.lagrange_nodes[2]; // right is vid, so node is lag[2]

            if (v_chart.is_cone) {
              // cone case
              m.insert(dep_node_id, indep_node_ids[0]) = 1;
              constrained_row_ids.push_back(dep_node_id);
            } else {
              Eigen::Vector2d u_im =
                  one_ring_uv_positions_map[e_chart.left_vertex_index] -
                  Eigen::Vector2d(0, 0);
              // Eigen::Vector2d u_im = e_chart.left_global_uv_position -
              //                        e_chart.right_global_uv_position;
              Eigen::Vector2d U_ijm = U_ijik_inv * u_im;

              // p_im = (1-Umj-Umk)*pi + Umj*pij + Umk*pik
              m.insert(dep_node_id, indep_node_ids[0]) =
                  1 - U_ijm[0] - U_ijm[1];
              m.insert(dep_node_id, indep_node_ids[1]) = U_ijm[0];
              m.insert(dep_node_id, indep_node_ids[2]) = U_ijm[1];
              constrained_row_ids.push_back(dep_node_id);
            }
            processed_id[e_chart.left_vertex_index] = true;
          }
          if (e_chart.right_vertex_index != vid &&
              processed_id.find(e_chart.right_vertex_index) ==
                  processed_id.end()) {

            // get node id
            auto dep_node_id =
                e_chart.lagrange_nodes[1]; // left is vid, so node is lag[1]

            if (v_chart.is_cone) {
              // cone case
              m.insert(dep_node_id, indep_node_ids[0]) = 1;
              constrained_row_ids.push_back(dep_node_id);
            } else {
              Eigen::Vector2d u_im =
                  one_ring_uv_positions_map[e_chart.right_vertex_index] -
                  Eigen::Vector2d(0, 0);
              // Eigen::Vector2d u_im = e_chart.right_global_uv_position -
              //                        e_chart.left_global_uv_position;
              Eigen::Vector2d U_ijm = U_ijik_inv * u_im;

              // p_im = (1-Umj-Umk)*pi + Umj*pij + Umk*pik
              m.insert(dep_node_id, indep_node_ids[0]) =
                  1 - U_ijm[0] - U_ijm[1];
              m.insert(dep_node_id, indep_node_ids[1]) = U_ijm[0];
              m.insert(dep_node_id, indep_node_ids[2]) = U_ijm[1];
              constrained_row_ids.push_back(dep_node_id);
            }
            processed_id[e_chart.right_vertex_index] = true;
          }
        }
      }
    }

    if (!v_chart.is_boundary) {
      assert(processed_id.size() == v_one_ring.size());
    } else {
      assert(processed_id.size() == v_one_ring.size() + 1);
    }
  }
}

void CloughTocherSurface::Ci_internal_ind2dep_1(
    Eigen::SparseMatrix<double> &m, std::vector<int64_t> &constrained_row_ids) {
  const auto &f_charts = m_affine_manifold.m_face_charts;

  for (size_t fid = 0; fid < f_charts.size(); ++fid) {
    const auto &f_chart = f_charts[fid];
    const auto &node_ids = f_chart.lagrange_nodes;

    // p0c = (p0 + p01 + p02) / 3
    Eigen::SparseVector<double> p0 = m.row(node_ids[0]);
    Eigen::SparseVector<double> p01 = m.row(node_ids[3]);
    Eigen::SparseVector<double> p02 = m.row(node_ids[8]);

    // m.row(node_ids[12]) = (p0 + p01 + p02) / 3.0;
    Eigen::SparseVector<double> p0c = (p0 + p01 + p02) / 3.0;
    assign_spvec_to_spmat_row(m, p0c, node_ids[12]);

    constrained_row_ids.push_back(node_ids[12]);

    // p1c = (p1 + p12 + p10) / 3
    Eigen::SparseVector<double> p1 = m.row(node_ids[1]);
    Eigen::SparseVector<double> p12 = m.row(node_ids[5]);
    Eigen::SparseVector<double> p10 = m.row(node_ids[4]);

    // m.row(node_ids[14]) = (p1 + p12 + p10) / 3.0;
    Eigen::SparseVector<double> p1c = (p1 + p12 + p10) / 3.0;
    assign_spvec_to_spmat_row(m, p1c, node_ids[14]);

    constrained_row_ids.push_back(node_ids[14]);

    // p2c = (p2 + p21 + p20) / 3
    Eigen::SparseVector<double> p2 = m.row(node_ids[2]);
    Eigen::SparseVector<double> p21 = m.row(node_ids[6]);
    Eigen::SparseVector<double> p20 = m.row(node_ids[7]);

    // m.row(node_ids[16]) = (p2 + p21 + p20) / 3.0;
    Eigen::SparseVector<double> p2c = (p2 + p21 + p20) / 3.0;
    assign_spvec_to_spmat_row(m, p2c, node_ids[16]);

    constrained_row_ids.push_back(node_ids[16]);
  }
}

std::array<int64_t, 7> N_helper(const int lid1, const int lid2) {
  switch (lid1 * 10 + lid2) {
  case 1:
    // 01
    return {{0, 1, 3, 4, 12, 14, 9}};
  case 2:
    // 02
    return {{0, 2, 8, 7, 12, 16, 11}};
  case 10:
    // 10
    return {{1, 0, 4, 3, 14, 12, 9}};
  case 12:
    // 12
    return {{1, 2, 5, 6, 14, 16, 10}};
  case 20:
    // 20
    return {{2, 0, 7, 8, 16, 12, 11}};
  case 21:
    // 21
    return {{2, 1, 6, 5, 16, 14, 10}};
  }

  return {{-1, -1, -1, -1, -1, -1, -1}};
}

void CloughTocherSurface::Ci_midpoint_ind2dep(
    Eigen::SparseMatrix<double> &m, std::vector<int64_t> &constrained_row_ids) {
  Eigen::Matrix<double, 5, 7> K_N;
  K_N << 1, 0, 0, 0, 0, 0, 0, // p0
      0, 1, 0, 0, 0, 0, 0,    // p1
      -3, 0, 3, 0, 0, 0, 0,   // d01
      0, -3, 0, 3, 0, 0, 0,   // d10
      -3. / 8., -3. / 8., -9. / 8., -9. / 8., 3. / 4., 3. / 4.,
      3. / 2.; // h01 redundant

  const Eigen::Matrix<double, 5, 1> c_e = c_e_m();
  const Eigen::Matrix<double, 7, 1> c_hij = c_hij_m();

  const auto &e_charts = m_affine_manifold.m_edge_charts;
  const auto &F = m_affine_manifold.get_faces();

  for (size_t eid = 0; eid < e_charts.size(); ++eid) {
    const auto &e_chart = e_charts[eid];
    if (e_chart.is_boundary) {
      // skip boundary edges
      continue;
    }

    const auto &fid_top = e_chart.top_face_index;
    const auto &fid_bot = e_chart.bottom_face_index;
    const auto &f_top = m_affine_manifold.m_face_charts[fid_top];
    const auto &f_bot = m_affine_manifold.m_face_charts[fid_bot];

    // v_pos and v_pos'
    const Eigen::Vector2d &v0_pos = e_chart.left_vertex_uv_position;
    const Eigen::Vector2d &v1_pos = e_chart.right_vertex_uv_position;
    const Eigen::Vector2d &v2_pos = e_chart.top_vertex_uv_position;

    const Eigen::Vector2d &v0_pos_prime = e_chart.right_vertex_uv_position;
    const Eigen::Vector2d &v1_pos_prime = e_chart.left_vertex_uv_position;
    const Eigen::Vector2d &v2_pos_prime = e_chart.bottom_vertex_uv_position;

    // u_ij and u_ij'
    const Eigen::Vector2d u_01 = v1_pos - v0_pos;
    const Eigen::Vector2d u_02 = v2_pos - v0_pos;
    const Eigen::Vector2d u_12 = v2_pos - v1_pos;

    std::cout << "u_01: " << u_01 << std::endl;

    const Eigen::Vector2d u_01_prime = v1_pos_prime - v0_pos_prime;
    const Eigen::Vector2d u_02_prime = v2_pos_prime - v0_pos_prime;
    const Eigen::Vector2d u_12_prime = v2_pos_prime - v1_pos_prime;

    // m01 and m01_prime
    const Eigen::Vector2d m_01 = (u_02 + u_12) / 2.0;
    const Eigen::Vector2d m_01_prime = (u_02_prime + u_12_prime) / 2.0;

    // u_01_prep u_01_prep_prime
    Eigen::Vector2d u_01_prep(-u_01[1], u_01[0]);
    Eigen::Vector2d u_01_prep_prime(-u_01_prime[1], u_01_prime[0]);

    // compute M_N and k_N
    Eigen::Matrix<double, 1, 7> M_N =
        (m_01.dot(u_01.normalized())) / u_01.norm() * c_e.transpose() * K_N;
    auto k_N = m_01.dot(u_01_prep.normalized());
    Eigen::Matrix<double, 1, 7> M_N_prime =
        (m_01_prime.dot(u_01_prime.normalized())) / u_01_prime.norm() *
        c_e.transpose() * K_N;
    // Eigen::Matrix<double, 1, 7> M_N_prime =
    //     (m_01_prime.dot(u_01.normalized())) / u_01_prime.norm() *
    //     c_e.transpose() * K_N;
    auto k_N_prime = m_01_prime.dot(u_01_prep_prime.normalized());

    // std::cout << c_hij.rows() << " " << c_hij.cols() << std::endl;
    // std::cout << M_N.rows() << " " << M_N.cols() << std::endl;

    // compute CM
    Eigen::Matrix<double, 1, 7> CM = c_hij - M_N.transpose(); // 7 x 1
    Eigen::Matrix<double, 1, 7> CM_prime = c_hij - M_N_prime.transpose();

    // get local indices in T and T'
    int lid1_top = -1;
    int lid2_top = -1;
    int lid1_bot = -1;
    int lid2_bot = -1;
    for (int i = 0; i < 3; ++i) {
      if (F.row(fid_top)[i] == e_chart.left_vertex_index) {
        lid1_top = i;
      }
      if (F.row(fid_top)[i] == e_chart.right_vertex_index) {
        lid2_top = i;
      }
      if (F.row(fid_bot)[i] == e_chart.right_vertex_index) {
        lid1_bot = i;
      }
      if (F.row(fid_bot)[i] == e_chart.left_vertex_index) {
        lid2_bot = i;
      }
    }

    assert(lid1_top > -1);
    assert(lid1_bot > -1);
    assert(lid2_top > -1);
    assert(lid2_bot > -1);

    // get N_full, N, N_prime
    const auto &node_ids_top = N_helper(lid1_top, lid2_top);
    const auto &node_ids_bot = N_helper(lid1_bot, lid2_bot);

    std::array<int64_t, 7> N;
    std::array<int64_t, 7> N_prime;
    for (int i = 0; i < 7; ++i) {
      N[i] = f_top.lagrange_nodes[node_ids_top[i]];
      N_prime[i] = f_bot.lagrange_nodes[node_ids_bot[i]];
    }

    assert(N[0] == N_prime[1]);
    assert(N[1] == N_prime[0]);
    assert(N[2] == N_prime[3]);
    assert(N[3] == N_prime[2]);

    auto p_0 = m.row(N[0]);
    auto p_1 = m.row(N[1]);
    auto p_01 = m.row(N[2]);
    auto p_10 = m.row(N[3]);
    auto p_0c = m.row(N[4]);
    auto p_1c = m.row(N[5]);

    auto p_0_prime = p_1;
    auto p_1_prime = p_0;
    auto p_01_prime = p_10;
    auto p_10_prime = p_01;

    auto p_0c_prime = m.row(N_prime[4]);
    auto p_1c_prime = m.row(N_prime[5]);

    // assign p_01_c as indep
    m.insert(N[6], N[6]) = 1;
    constrained_row_ids.push_back(N[6]);

    auto p_01_c = m.row(N[6]);

    Eigen::SparseVector<double> p_01_c_prime =
        (k_N_prime * (CM[0] * p_0 + CM[1] * p_1 + CM[2] * p_01 + CM[3] * p_10 +
                      CM[4] * p_0c + CM[5] * p_1c + CM[6] * p_01_c) +
         k_N * (CM_prime[0] * p_0_prime + CM_prime[1] * p_1_prime +
                CM_prime[2] * p_01_prime + CM_prime[3] * p_10_prime +
                CM_prime[4] * p_0c_prime + CM_prime[5] * p_1c_prime)) /
        (-k_N * CM_prime[6]);

    // // assign p_01_c_prime
    // m.row(N_prime[6]) = p_01_c_prime;
    assign_spvec_to_spmat_row(m, p_01_c_prime, N_prime[6]);
    constrained_row_ids.push_back(N_prime[6]);
  }
}

void CloughTocherSurface::Ci_internal_ind2dep_2(
    Eigen::SparseMatrix<double> &m) {
  const auto &f_charts = m_affine_manifold.m_face_charts;

  for (size_t fid = 0; fid < f_charts.size(); ++fid) {
    const auto &f_chart = f_charts[fid];
    const auto &node_ids = f_chart.lagrange_nodes;

    // pc0 = (p0c + p01^c + p20^c) / 3
    Eigen::SparseVector<double> p0c = m.row(node_ids[12]);
    Eigen::SparseVector<double> p01_c = m.row(node_ids[9]);
    Eigen::SparseVector<double> p20_c = m.row(node_ids[11]);

    Eigen::SparseVector<double> pc0 = (p0c + p01_c + p20_c) / 3.0;
    // m.row(node_ids[13]) = pc0;

    assign_spvec_to_spmat_row(m, pc0, node_ids[13]);

    // pc1 = (p1c + p12^c + p01^c) / 3
    Eigen::SparseVector<double> p1c = m.row(node_ids[14]);
    Eigen::SparseVector<double> p12_c = m.row(node_ids[10]);
    // Eigen::SparseMatrix<double> p01_c = m.row(node_ids[9]);

    Eigen::SparseVector<double> pc1 = (p1c + p12_c + p01_c) / 3.0;
    // m.row(node_ids[15]) = pc1;

    assign_spvec_to_spmat_row(m, pc1, node_ids[15]);

    // pc2 = (p2c + p20^c + p12^c) / 3
    Eigen::SparseVector<double> p2c = m.row(node_ids[16]);
    // Eigen::SparseMatrix<double> p20_c = m.row(node_ids[11]);
    // Eigen::SparseMatrix<double> p12_c = m.row(node_ids[10]);

    Eigen::SparseVector<double> pc2 = (p2c + p20_c + p12_c) / 3.0;
    // m.row(node_ids[17]) = pc2;

    assign_spvec_to_spmat_row(m, pc2, node_ids[17]);

    // pc = (pc0 + pc1 + pc2) / 3
    // m.row(node_ids[18]) = (pc0 + pc1 + pc2) / 3.0;
    Eigen::SparseVector<double> pc = (pc0 + pc1 + pc2) / 3.0;
    assign_spvec_to_spmat_row(m, pc, node_ids[18]);
  }
}