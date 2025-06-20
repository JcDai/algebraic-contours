add_subdirectory(autogen)

add_library(CloughTocherPatchLib
  clough_tocher_patch.hpp
  clough_tocher_patch.cpp
  clough_tocher_matrices.cpp
  clough_tocher_surface.cpp
  clough_tocher_constraint_matrices.hpp
  clough_tocher_constraint_matrices.cpp
  optimize_clough_tocher.cpp
  autogen/clough_tocher_autogen_matrix_coeffs.hpp
  autogen/clough_tocher_autogen_constraint_matrices.hpp
)
target_include_directories(CloughTocherPatchLib PUBLIC .)
target_link_libraries(CloughTocherPatchLib PUBLIC
  QuadraticContoursCoreLib
  QuadraticSplineSurfaceLib
)
target_compile_definitions(CloughTocherPatchLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
  # SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF
)

target_compile_options(CloughTocherPatchLib PRIVATE
  -Wall -Wpedantic -Wextra
)