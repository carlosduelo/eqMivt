/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_RAYCASTER_GRID_CUDA_H
#define EQ_MIVY_RAYCASTER_GRID_CUDA_H

#include "typedef.h"

#include <cuda_runtime.h>


namespace eqMivt
{
	void rayCasterGrid_CUDA(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, float iso, visibleCube_t * cube, int * indexCube, int3 cubeDim, int3 cubeInc, float maxHeight, float * pixelBuffer, float * xGrid, float * yGrid, float * zGrid, int3 realDim, float * r, float * g, float * b, cudaStream_t stream);

	void rayCasterGrid_Cubes_CUDA(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, float iso, visibleCube_t * cube, int * indexCube, int3 cubeDim, int3 cubeInc, float maxHeight, float * pixelBuffer, float * xGrid, float * yGrid, float * zGrid, int3 realDim, float * r, float * g, float * b, cudaStream_t stream);
}

#endif
