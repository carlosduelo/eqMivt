# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}. 

set(MIVT_SOURCE 
		memoryCheck.cpp
		hdf5File.cpp
		fileFactory.cpp
		linkedList.cpp
		cubeCacheCPU.cpp
		cubeCacheGPU.cpp
		cubeCache.cpp
		cacheManager.cpp
		octree.cpp
		octreeManager.cpp
		node.cpp
		render.cpp
		pipe.cpp
		channel.cpp
		rayCaster.cpp
		error.cpp
		initData.cpp
		localInitData.cpp
		frameData.cpp
		config.cpp
		view.cpp
		window.cpp
		eqMivt.cpp
)

set(MIVT_CUDA_SOURCE ray-casting-sphere.cu octree_CUDA.cu rayCaster_CUDA.cu octreeGrid_CUDA.cu rayCasterGrid_CUDA.cu)

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

set(LIBS_NEEDED ${EQUALIZER_LIBRARIES} ${CUDA_LIBRARIES} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${FREEIMAGE_LIBRARIES} ${OPENGL_gl_LIBRARY} dl GL) 

add_executable(mivt ${MIVT_SOURCE} ${MIVT_CUDA_OBJECTS} main.cpp) 
target_link_libraries(mivt ${LIBS_NEEDED}) 


set(OCTREE_CONSTRUCTOR_SOURCE 
		fileFactory.cpp
		hdf5File.cpp
		octreeConstructor.cpp
		memoryCheck.cpp
	)

# COMPILING CUDA OBJECTS
set(MIVT_OC_CUDA_SOURCE octreeConstructor_CUDA.cu)
cuda_compile(MIVT_OC_CUDA_OBJECTS ${MIVT_OC_CUDA_SOURCE})
add_executable(octreeConstructor ${OCTREE_CONSTRUCTOR_SOURCE} ${MIVT_OC_CUDA_OBJECTS} createOctree.cpp) 
target_link_libraries(octreeConstructor ${LIBS_NEEDED}) 

add_executable(printOctree  printOctree.cpp) 
target_link_libraries(printOctree ${LIBS_NEEDED}) 
