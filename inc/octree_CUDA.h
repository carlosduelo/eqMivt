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
	void resetStateOctree(cudaStream_t stream, int * GstackActual, index_node_t * GstackIndex, int * GstackLevel, int maxRays);

	void getBoxIntersectedOctree(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float* rays, int finalLevel, int numElements, int * stackActual, index_node_t * stackIndex, int * stackLevel, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, cudaStream_t stream);
}
#endif
