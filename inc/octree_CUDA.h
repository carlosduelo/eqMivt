/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef _EQ_MIVT_OCTREE_CUDA_H_
#define _EQ_MIVT_OCTREE_CUDA_H_

#include "typedef.h"

namespace eqMivt
{
	bool Create_Octree(index_node_t ** octreeCPU, int * sizesCPU, int maxLevel, index_node_t *** octree, index_node_t ** memoryGPU, int ** sizes); 

	bool Destroy_Octree(index_node_t ** octree, index_node_t * memoryGPU, int * sizes); 

	void getBoxIntersectedOctree(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, int numElements, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, int * indexVisibleCubesGPU, int * indexVisibleCubesCPU, cudaStream_t stream);
}
#endif
