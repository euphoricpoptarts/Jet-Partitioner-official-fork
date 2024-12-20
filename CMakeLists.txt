cmake_minimum_required(VERSION 3.23)
project(jetpartition LANGUAGES CXX VERSION 1.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# GKlib is a dependency of Metis when built from its github repo
# Unfortunately there is no simple way to avoid also linking GKlib in this case
# Older distributions of metis do not create this dependency
SET(LINK_GKLIB False CACHE BOOL "Newer Metis distributions require us to link GKlib")

find_package(KokkosKernels REQUIRED)
add_compile_options(-Wall -Wextra -Wshadow)

# libjet
add_library(jet STATIC jet.cpp)
target_sources(jet PUBLIC
  FILE_SET HEADERS
  FILES jet.h jet_defs.h jet_config.h ExperimentLoggerUtil.hpp
)

if(DEFINED METIS_DIR)
# This is used by the build script
# to avoid putting metis and gklib in the global path
include_directories(${METIS_DIR}/include)
# downstream targets need to know about these link directories
target_link_directories(jet INTERFACE ${METIS_DIR}/lib)
endif()

# link libjet (for downstream library consumers)
# no linking actually occurs for libjet itself
target_link_libraries(jet PUBLIC Kokkos::kokkos Kokkos::kokkoskernels)
target_link_libraries(jet PRIVATE metis)
if(LINK_GKLIB)
target_link_libraries(jet PRIVATE GKlib)
endif()

# jet executables
add_executable(jet_ex driver.cpp)
set_property(TARGET jet_ex PROPERTY OUTPUT_NAME jet)
add_executable(jet4 driver.cpp)
add_executable(jet2 driver.cpp)
add_executable(jet_host driver.cpp)
add_executable(jet_import import_coarse.cpp)
add_executable(jet_export driver.cpp)
add_executable(jet_serial driver.cpp)
add_executable(pstat part_eval.cpp)

# compile definitions to set exe behavior
target_compile_definitions(jet4 PUBLIC FOUR9)
target_compile_definitions(jet2 PUBLIC TWO9)
target_compile_definitions(jet_host PUBLIC HOST)
target_compile_definitions(jet_import PUBLIC HOST)
target_compile_definitions(jet_export PUBLIC HOST EXP)
target_compile_definitions(jet_serial PUBLIC SERIAL)

# link executables
target_link_libraries(jet_import Kokkos::kokkos Kokkos::kokkoskernels)
target_link_libraries(pstat Kokkos::kokkos Kokkos::kokkoskernels)
foreach(prog jet_ex jet4 jet2 jet_host jet_export jet_serial)
    target_link_libraries(${prog} jet)
endforeach(prog)

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