# GKlib is a dependency of Metis when built from its github repo
# Unfortunately there is no simple way to avoid also linking GKlib in this case
# Older distributions of metis do not create this dependency
SET(LINK_GKLIB False CACHE BOOL "Newer Metis distributions require us to link GKlib")

# libjet
add_library(jet STATIC jet.cpp)
target_sources(jet PUBLIC
  FILE_SET HEADERS
  BASE_DIRS ${CMAKE_SOURCE_DIR}/header
  FILES ${CMAKE_SOURCE_DIR}/header/jet.h ${CMAKE_SOURCE_DIR}/header/jet_defs.h ${CMAKE_SOURCE_DIR}/header/jet_config.h ${CMAKE_SOURCE_DIR}/header/experiment_data.hpp)
target_include_directories(jet PRIVATE ${CMAKE_SOURCE_DIR}/header)

if(DEFINED METIS_DIR)
# This is used by the build script
# to avoid putting metis and gklib in the global path
target_include_directories(jet PRIVATE ${METIS_DIR}/include)
# downstream targets need to know about these link directories
target_link_directories(jet INTERFACE ${METIS_DIR}/lib)
endif()

# link libjet (for downstream library consumers)
# no linking actually occurs for libjet itself
find_package(KokkosKernels REQUIRED)
target_link_libraries(jet PUBLIC Kokkos::kokkos Kokkos::kokkoskernels)
target_link_libraries(jet PRIVATE metis)
if(LINK_GKLIB)
target_link_libraries(jet PRIVATE GKlib)
endif()

# install jetlib and create cmake package
install(TARGETS jet
EXPORT jetTargets
FILE_SET HEADERS DESTINATION include
ARCHIVE DESTINATION lib)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/jet-config-version.cmake
  COMPATIBILITY AnyNewerVersion
)

# install the cmake package
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/jet-config-version.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/jetConfig.cmake
  DESTINATION lib/cmake/jet
)
install(EXPORT jetTargets
  FILE
    jetTargets.cmake
  NAMESPACE
    jet::
  DESTINATION
    lib/cmake/jet
)