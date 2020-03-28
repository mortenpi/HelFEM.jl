project(helfem)

cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")

message(STATUS "LIBHELFEM_SRC ${LIBHELFEM_SRC}")
message(STATUS "LIBXC ${LIBXC}")
message(STATUS "ARMADILLO ${ARMADILLO}")
message(STATUS "HDF5 ${HDF5}")

include_directories("${LIBXC}/include" "${ARMADILLO}/include" "${HDF5}/include")
set(CMAKE_INSTALL_RPATH "${HDF5}/lib:${ARMADILLO}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

message(STATUS "Found JlCxx at ${JlCxx_location}")

add_library(helfem SHARED
  ${LIBHELFEM_SRC}/general/polynomial.cpp
  ${LIBHELFEM_SRC}/general/polynomial_basis.cpp
  ${LIBHELFEM_SRC}/general/gaunt.cpp
  ${LIBHELFEM_SRC}/general/chebyshev.cpp
  ${LIBHELFEM_SRC}/general/diis.cpp
  ${LIBHELFEM_SRC}/general/lbfgs.cpp
  ${LIBHELFEM_SRC}/general/utils.cpp
  ${LIBHELFEM_SRC}/general/spherical_harmonics.cpp
  ${LIBHELFEM_SRC}/general/timer.cpp
  ${LIBHELFEM_SRC}/general/elements.cpp
  ${LIBHELFEM_SRC}/general/lobatto.cpp
  ${LIBHELFEM_SRC}/general/angular.cpp
  ${LIBHELFEM_SRC}/general/scf_helpers.cpp
  ${LIBHELFEM_SRC}/general/lcao.cpp
  ${LIBHELFEM_SRC}/general/gsz.cpp
  ${LIBHELFEM_SRC}/general/sap.cpp
  ${LIBHELFEM_SRC}/general/dftfuncs.cpp
  ${LIBHELFEM_SRC}/general/checkpoint.cpp
  ${LIBHELFEM_SRC}/atomic/basis.cpp
  ${LIBHELFEM_SRC}/atomic/quadrature.cpp
  ${LIBHELFEM_SRC}/atomic/dftgrid.cpp
  ${LIBHELFEM_SRC}/atomic/erfc_expn.cpp
  ${LIBHELFEM_SRC}/sadatom/basis.cpp
  ${LIBHELFEM_SRC}/sadatom/dftgrid.cpp
  ${LIBHELFEM_SRC}/sadatom/solver.cpp
  ${LIBHELFEM_SRC}/sadatom/configurations.cpp
  ${LIBHELFEM_SRC}/general/dftfuncs.cpp
  ${LIBHELFEM_SRC}/diatomic/basis.cpp
  ${LIBHELFEM_SRC}/diatomic/quadrature.cpp
  ${LIBHELFEM_SRC}/diatomic/dftgrid.cpp
  ${LIBHELFEM_SRC}/diatomic/twodquadrature.cpp
  ${LIBHELFEM_SRC}/general/model_potential.cpp
  helfem.cc)
target_link_libraries(helfem
  JlCxx::cxxwrap_julia
  "${LIBXC}/lib/libxc.a"
  "${HDF5}/lib/libhdf5.so.103.0.0"
  "${ARMADILLO}/lib/libarmadillo.so"
)

install(TARGETS helfem
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)
