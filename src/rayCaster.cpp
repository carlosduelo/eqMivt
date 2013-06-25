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
#define VectorToInt3(v) make_int3((v).x(), (v).y(), (v).z())

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

void rayCaster::setMaxHeight(float maxHeight)
{
	_maxHeight = maxHeight;
}

void rayCaster::increaseStep()
{
	_step += 0.01f;
}

void rayCaster::decreaseStep()
{
	_step += _step == 0.01f ? 0.0f : 0.01f;
}

void rayCaster::render(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, visibleCube_t * cube,int * indexCube, int3 cubeDim, int3 cubeInc, float * pixelBuffer, float * xGrid, float * yGrid, float * zGrid, vmml::vector<3, int> realDim, cudaStream_t stream)
{
	rayCaster_CUDA(VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH, numRays, levelO, levelC, nLevel, _iso, cube, indexCube, cubeDim, cubeInc, _maxHeight, pixelBuffer, xGrid, yGrid, zGrid, VectorToInt3(realDim), stream);
}

void rayCaster::renderCubes(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, visibleCube_t * cube,int * indexCube, int3 cubeDim, int3 cubeInc, float * pixelBuffer, float * xGrid, float * yGrid, float * zGrid, vmml::vector<3, int> realDim, cudaStream_t stream)
{
	rayCaster_Cubes_CUDA(VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH, numRays, levelO, levelC, nLevel, _iso, cube, indexCube, cubeDim, cubeInc, _maxHeight, pixelBuffer, xGrid, yGrid, zGrid, VectorToInt3(realDim), stream);
}


}
