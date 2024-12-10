cmake_minimum_required(VERSION 3.18)
project(jetpartition CXX)
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

# This is used by the build script
# to avoid putting metis and gklib in the global path
include_directories(${METIS_DIR}/include)
link_directories(${METIS_DIR}/lib)

add_library(jet jet.cpp)
add_executable(jet_ex driver.cpp)
set_property(TARGET jet_ex PROPERTY OUTPUT_NAME jet)
add_executable(jet4 driver.cpp)
add_executable(jet2 driver.cpp)
add_executable(jet_host driver.cpp)
add_executable(jet_import import_coarse.cpp)
add_executable(jet_export driver.cpp)
add_executable(jet_serial driver.cpp)
add_executable(pstat part_eval.cpp)

target_compile_definitions(jet4 PUBLIC FOUR9)
target_compile_definitions(jet2 PUBLIC TWO9)
target_compile_definitions(jet_host PUBLIC HOST)
target_compile_definitions(jet_import PUBLIC HOST)
target_compile_definitions(jet_export PUBLIC HOST EXP)
target_compile_definitions(jet_serial PUBLIC SERIAL)
foreach(prog jet_ex jet4 jet2 jet_host jet_import jet_export jet_serial pstat jet)
    target_link_libraries(${prog} Kokkos::kokkos Kokkos::kokkoskernels)
endforeach(prog)
target_link_libraries(jet metis)
if(LINK_GKLIB)
target_link_libraries(jet GKlib)
endif()
foreach(prog jet_ex jet4 jet2 jet_host jet_export jet_serial)
    target_link_libraries(${prog} jet)
endforeach(prog)