# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}. 

set(MIVT_SOURCE 
		error.cpp
		initData.cpp
		localInitData.cpp
		frameData.cpp
		config.cpp
		node.cpp
		pipe.cpp
		channel.cpp
		eqMivt.cpp
)

set(MIVT_CUDA_SOURCE ray-casting-sphere.cu )

# SETTING CUDA PARAMETERS
list(APPEND CUDA_NVCC_FLAGS -arch=sm_13)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	list(APPEND CUDA_NVCC_FLAGS -G)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O0)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall)
else()
	list(APPEND CUDA_NVCC_FLAGS -O3)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O3)
endif()

# COMPILING CUDA OBJECTS
cuda_compile(MIVT_CUDA_OBJECTS ${MIVT_CUDA_SOURCE})
#message(STATUS ${MIVT_CUDA_OBJECTS})

set(LIBS_NEEDED ${EQUALIZER_LIBRARIES} GL ${CUDA_LIBRARIES})# ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ) 

add_executable(mivt ${MIVT_SOURCE} ${MIVT_CUDA_OBJECTS} main.cpp) 
target_link_libraries(mivt ${LIBS_NEEDED}) 
