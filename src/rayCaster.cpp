#include "rayCaster.h"

#include "rayCaster_CUDA.h"

#include <iostream>
#include <fstream>

namespace eqMivt
{

rayCaster::rayCaster()
{
	_iso    = 0.0f; 
	_step	= 0.5f;
}

rayCaster::~rayCaster()
{
}

void rayCaster::setIsosurface(float isosurface)
{
	_iso = isosurface;
}

void rayCaster::increaseStep()
{
	_step += 0.01f;
}

void rayCaster::decreaseStep()
{
	_step += _step == 0.01f ? 0.0f : 0.01f;
}

void rayCaster::render(float3 origin, float * rays, int numRays, int levelO, int levelC, int nLevel, visibleCube_t * cube, int3 cubeDim, int3 cubeInc, float * pixelBuffer, cudaStream_t stream)
{
	rayCaster_CUDA(origin, rays, numRays, levelO, levelC, nLevel, _iso, cube, cubeDim, cubeInc, pixelBuffer, stream);
}



}
