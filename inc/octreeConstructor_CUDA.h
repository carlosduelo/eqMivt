/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef _EQ_MIVT_CONSTRUCTOR_OCTREE_CUDA_H_
#define _EQ_MIVT_CONSTRUCTOR_OCTREE_CUDA_H_

#include "typedef.h"

namespace eqMivt
{
	size_t	octreeConstructorGetFreeMemory();	
	float * octreeConstructorCreateCube(int cubeDim);
	bool	octreeConstructorCopyCube(float * cubeGPU, float * cube, int cubeDim);
	bool	octreeConstructorCopyCube3D(float * cubeGPU, float * cube, int cubeDimC, int cubeDimG, int xoffset, int yoffset, int zoffset);
	index_node_t * octreeConstructorCreateResult(int size);
	bool	octreeConstructorCopyResult(index_node_t * cpuResult, index_node_t * gpuResult, int size);
	void	octreeConstructorDestroyCube(float * cube);
	void	octreeConstructorDestroyResult(index_node_t * result);
	void	octreeConstructorComputeCube(index_node_t * cubes, int size, index_node_t startID, float iso, float * cube, int nodeLevel, int nLevels, int dimNode, int cubeDim, int coorCubeStart[3]);
}
#endif /*_EQ_MIVT_CONSTRUCTOR_OCTREE_CUDA_H_*/
