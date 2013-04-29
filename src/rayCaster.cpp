/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/
#include "rayCaster.h"

#include "rayCaster_CUDA.h"

#include <iostream>
#include <fstream>

#define VectorToFloat3(v) make_float3((v).x(), (v).y(), (v).z())

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

void rayCaster::render(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, eq::Vector2f jitter, int numRays, int levelO, int levelC, int nLevel, visibleCube_t * cube, int3 cubeDim, int3 cubeInc, float * pixelBuffer, cudaStream_t stream)
{
	rayCaster_CUDA(VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH, make_float2(jitter.x(), jitter.y()), numRays, levelO, levelC, nLevel, _iso, cube, cubeDim, cubeInc, pixelBuffer, stream);
}



}
