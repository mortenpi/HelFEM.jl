project(helfem)

cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)

message(STATUS "LIBHELFEM_SRC ${LIBHELFEM_SRC}")
message(STATUS "LIBXC ${LIBXC}")
message(STATUS "ARMADILLO ${ARMADILLO}")
message(STATUS "HDF5 ${HDF5}")
message(STATUS "OPENBLAS ${OPENBLAS}")
message(STATUS "GSL ${GSL}")

add_definitions(-DARMA_64BIT_WORD -DARMA_DONT_USE_WRAPPER)
# This list of BLAS re-definitions is derived from the Armadillo Yggdrasil build script.
# Essentially, it seems that the OpenBLAS packaged with BinaryProvider appends _64 to all
# the symbols (possibly, to differentiate from the OpenBLAS symbols that Julia itself links
# against?). So, armadillo has to rename them all as well, and so has to everything that
# depends on Armadillo, since the Armadillo wrappers also get renamed.
#
# https://github.com/JuliaPackaging/Yggdrasil/blob/1906ffd7962d07e1d52e866291851a918faa1eaf/A/armadillo/build_tarballs.jl#L28-L38
add_definitions(-Dsasum=sasum_64 -Ddasum=dasum_64 -Dsnrm2=snrm2_64 -Ddnrm2=dnrm2_64 -Dsdot=sdot_64 -Dddot=ddot_64 -Dsgemv=sgemv_64 -Ddgemv=dgemv_64 -Dcgemv=cgemv_64 -Dzgemv=zgemv_64 -Dsgemm=sgemm_64 -Ddgemm=dgemm_64 -Dcgemm=cgemm_64 -Dzgemm=zgemm_64 -Dssyrk=ssyrk_64 -Ddsyrk=dsyrk_64 -Dcherk=cherk_64 -Dzherk=zherk_64 -Dcgbcon=cgbcon_64 -Dcgbsv=cgbsv_64 -Dcgbsvx=cgbsvx_64 -Dcgbtrf=cgbtrf_64 -Dcgbtrs=cgbtrs_64 -Dcgecon=cgecon_64 -Dcgees=cgees_64 -Dcgeev=cgeev_64 -Dcgeevx=cgeevx_64 -Dcgehrd=cgehrd_64 -Dcgels=cgels_64 -Dcgelsd=cgelsd_64 -Dcgemm=cgemm_64 -Dcgemv=cgemv_64 -Dcgeqrf=cgeqrf_64 -Dcgesdd=cgesdd_64 -Dcgesv=cgesv_64 -Dcgesvd=cgesvd_64 -Dcgesvx=cgesvx_64 -Dcgetrf=cgetrf_64 -Dcgetri=cgetri_64 -Dcgetrs=cgetrs_64 -Dcgges=cgges_64 -Dcggev=cggev_64 -Dcgtsv=cgtsv_64 -Dcgtsvx=cgtsvx_64 -Dcheev=cheev_64 -Dcheevd=cheevd_64 -Dcherk=cherk_64 -Dclangb=clangb_64 -Dclange=clange_64 -Dclanhe=clanhe_64 -Dclansy=clansy_64 -Dcpbtrf=cpbtrf_64 -Dcpocon=cpocon_64 -Dcposv=cposv_64 -Dcposvx=cposvx_64 -Dcpotrf=cpotrf_64 -Dcpotri=cpotri_64 -Dcpotrs=cpotrs_64 -Dctrcon=ctrcon_64 -Dctrsyl=ctrsyl_64 -Dctrtri=ctrtri_64 -Dctrtrs=ctrtrs_64 -Dcungqr=cungqr_64 -Ddasum=dasum_64 -Dddot=ddot_64 -Ddgbcon=dgbcon_64 -Ddgbsv=dgbsv_64 -Ddgbsvx=dgbsvx_64 -Ddgbtrf=dgbtrf_64 -Ddgbtrs=dgbtrs_64 -Ddgecon=dgecon_64 -Ddgees=dgees_64 -Ddgeev=dgeev_64 -Ddgeevx=dgeevx_64 -Ddgehrd=dgehrd_64 -Ddgels=dgels_64 -Ddgelsd=dgelsd_64 -Ddgemm=dgemm_64 -Ddgemv=dgemv_64 -Ddgeqrf=dgeqrf_64 -Ddgesdd=dgesdd_64 -Ddgesv=dgesv_64 -Ddgesvd=dgesvd_64 -Ddgesvx=dgesvx_64 -Ddgetrf=dgetrf_64 -Ddgetri=dgetri_64 -Ddgetrs=dgetrs_64 -Ddgges=dgges_64 -Ddggev=dggev_64 -Ddgtsv=dgtsv_64 -Ddgtsvx=dgtsvx_64 -Ddlahqr=dlahqr_64 -Ddlangb=dlangb_64 -Ddlange=dlange_64 -Ddlansy=dlansy_64 -Ddlarnv=dlarnv_64 -Ddnrm2=dnrm2_64 -Ddorgqr=dorgqr_64 -Ddpbtrf=dpbtrf_64 -Ddpocon=dpocon_64 -Ddposv=dposv_64 -Ddposvx=dposvx_64 -Ddpotrf=dpotrf_64 -Ddpotri=dpotri_64 -Ddpotrs=dpotrs_64 -Ddstedc=dstedc_64 -Ddsyev=dsyev_64 -Ddsyevd=dsyevd_64 -Ddsyrk=dsyrk_64 -Ddtrcon=dtrcon_64 -Ddtrevc=dtrevc_64 -Ddtrsyl=dtrsyl_64 -Ddtrtri=dtrtri_64 -Ddtrtrs=dtrtrs_64 -Dilaenv=ilaenv_64 -Dsasum=sasum_64 -Dsdot=sdot_64 -Dsgbcon=sgbcon_64 -Dsgbsv=sgbsv_64 -Dsgbsvx=sgbsvx_64 -Dsgbtrf=sgbtrf_64 -Dsgbtrs=sgbtrs_64 -Dsgecon=sgecon_64 -Dsgees=sgees_64 -Dsgeev=sgeev_64 -Dsgeevx=sgeevx_64 -Dsgehrd=sgehrd_64 -Dsgels=sgels_64 -Dsgelsd=sgelsd_64 -Dsgemm=sgemm_64 -Dsgemv=sgemv_64 -Dsgeqrf=sgeqrf_64 -Dsgesdd=sgesdd_64 -Dsgesv=sgesv_64 -Dsgesvd=sgesvd_64 -Dsgesvx=sgesvx_64 -Dsgetrf=sgetrf_64 -Dsgetri=sgetri_64 -Dsgetrs=sgetrs_64 -Dsgges=sgges_64 -Dsggev=sggev_64 -Dsgtsv=sgtsv_64 -Dsgtsvx=sgtsvx_64 -Dslahqr=slahqr_64 -Dslangb=slangb_64 -Dslange=slange_64 -Dslansy=slansy_64 -Dslarnv=slarnv_64 -Dsnrm2=snrm2_64 -Dsorgqr=sorgqr_64 -Dspbtrf=spbtrf_64 -Dspocon=spocon_64 -Dsposv=sposv_64 -Dsposvx=sposvx_64 -Dspotrf=spotrf_64 -Dspotri=spotri_64 -Dspotrs=spotrs_64 -Dsstedc=sstedc_64 -Dssyev=ssyev_64 -Dssyevd=ssyevd_64 -Dssyrk=ssyrk_64 -Dstrcon=strcon_64 -Dstrevc=strevc_64 -Dstrsyl=strsyl_64 -Dstrtri=strtri_64 -Dstrtrs=strtrs_64 -Dzgbcon=zgbcon_64 -Dzgbsv=zgbsv_64 -Dzgbsvx=zgbsvx_64 -Dzgbtrf=zgbtrf_64 -Dzgbtrs=zgbtrs_64 -Dzgecon=zgecon_64 -Dzgees=zgees_64 -Dzgeev=zgeev_64 -Dzgeevx=zgeevx_64 -Dzgehrd=zgehrd_64 -Dzgels=zgels_64 -Dzgelsd=zgelsd_64 -Dzgemm=zgemm_64 -Dzgemv=zgemv_64 -Dzgeqrf=zgeqrf_64 -Dzgesdd=zgesdd_64 -Dzgesv=zgesv_64 -Dzgesvd=zgesvd_64 -Dzgesvx=zgesvx_64 -Dzgetrf=zgetrf_64 -Dzgetri=zgetri_64 -Dzgetrs=zgetrs_64 -Dzgges=zgges_64 -Dzggev=zggev_64 -Dzgtsv=zgtsv_64 -Dzgtsvx=zgtsvx_64 -Dzheev=zheev_64 -Dzheevd=zheevd_64 -Dzherk=zherk_64 -Dzlangb=zlangb_64 -Dzlange=zlange_64 -Dzlanhe=zlanhe_64 -Dzlansy=zlansy_64 -Dzpbtrf=zpbtrf_64 -Dzpocon=zpocon_64 -Dzposv=zposv_64 -Dzposvx=zposvx_64 -Dzpotrf=zpotrf_64 -Dzpotri=zpotri_64 -Dzpotrs=zpotrs_64 -Dztrcon=ztrcon_64 -Dztrsyl=ztrsyl_64 -Dztrtri=ztrtri_64 -Dztrtrs=ztrtrs_64 -Dzungqr=zungqr_64)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location};${HDF5}/lib;${OPENBLAS}/lib;${GSL}/lib")
include_directories("${LIBXC}/include" "${ARMADILLO}/include" "${HDF5}/include" "${GSL}/include" "${LIBHELFEM_SRC}")
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
  "${OPENBLAS}/lib/libopenblas64_.so"
  "${GSL}/lib/libgsl.so"
)

install(TARGETS helfem
  LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
  RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
