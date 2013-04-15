# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}. 

cmake_minimum_required (VERSION 2.8)

project (mivt)

# Equalizer
find_package(Equalizer 1.5.0 REQUIRED)
if (EQUALIZER_FOUND)
	include_directories(${EQUALIZER_INCLUDE_DIRS})
else()
	message(ERROR "EQUALIZER NOT FOUND")
endif()

set(CMAKE_BUILD_TYPE Debug)

# BOOST 
set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost 1.36.0 COMPONENTS system filesystem program_options)
if (Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
else()
	message(ERROR "Boost NOT FOUND")
endif()

# HDF5
#find_package(HDF5 REQUIRED)
#if (HDF5_FOUND)
#	include_directories(HDF5_INCLUDE_DIRS)
#else()
#	message(ERROR "HDF5 NOT FOUND")
#endif()

# CUDA
find_package(CUDA 5.0 REQUIRED)
if (CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRS})
	cuda_include_directories(${mivt_SOURCE_DIR}/inc)
else()
	message(ERROR "CUDA NOT FOUND")
endif()


# TOOLS
#include_directories(${TCLAP_INDCLUDE_DIR})
#message(STATUS ${TCLAP_INDCLUDE_DIR})
#include_directories(${mivt_SOURCE_DIR}/tools/tclap/include)


# MIVT
include_directories(${mivt_SOURCE_DIR}/inc)
add_subdirectory(${mivt_SOURCE_DIR}/src)
