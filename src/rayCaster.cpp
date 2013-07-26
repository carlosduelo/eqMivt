/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/
#include "rayCaster.h"

#include "rayCaster_CUDA.h"
#include "rayCasterGrid_CUDA.h"

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
	_grid   = false;
	_r = 0;
	_g = 0;
	_b = 0;
}

rayCaster::~rayCaster()
{
	rayCasterDestroyColors(_r);
}

void rayCaster::setIsosurface(float isosurface)
{
	_iso = isosurface;
}

void rayCaster::setMaxHeight(float maxHeight)
{
	_maxHeight = maxHeight;
}
void rayCaster::setUseGrid(bool grid)
{
	_grid = grid;
}
void rayCaster::setOffset(vmml::vector<3, int> offset)
{
	_offset = offset;
}

bool rayCaster::setColor(float * c)
{
	bool result = true;
	if (_r==0 && _g== 0 && _b == 0)
	{
		result = rayCasterCreateColors(&_r, &_g, &_b);
		if (c == 0)
		{
			float * cc = new float[3*NUM_COLORS + 3];
			for(int p=0; p<NUM_COLORS; p++)
				cc[p] = 1.0f;

			for(int p=0; p<64; p++)
				cc[(NUM_COLORS+1) + p] = 0.0f;

			float dc = 1.0f/((float)NUM_COLORS - 60.0f);
			int k = 1;
			for(int p=64; p<NUM_COLORS; p++)
			{
				cc[(NUM_COLORS+1) + p] = (float)k*dc; 
				k++;
			}

			for(int p=0; p<192; p++)
				cc[2*(NUM_COLORS+1) + p] = 0.0f;

			dc = 1.0f/100.0f;
			k=1;
			for(int p=192; p<NUM_COLORS; p++)
			{
				cc[(2*(NUM_COLORS+1)) + p] = (float)k*dc; 
				k++;
			}
		
			#ifdef DEBUG
			//BACKGROUND COLOR
			cc[NUM_COLORS] = rand()/(float)RAND_MAX;
			cc[2*NUM_COLORS+1] = rand()/(float)RAND_MAX;
			cc[3*NUM_COLORS+2] = rand()/(float)RAND_MAX;
			#else
			cc[NUM_COLORS] = 1.0f;
			cc[2*NUM_COLORS+1] = 1.0f;
			cc[3*NUM_COLORS+2] = 1.0f;
			#endif

			result &= rayCasterCopyColors(cc, _r);
			delete[] cc;
		}
		else
		{
			result &= rayCasterCopyColors(c, _r);
		}
	}
	else
		return false;

	return result;
}


void rayCaster::render(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, visibleCube_t * cube,int * indexCube, int3 cubeDim, int3 cubeInc, float * pixelBuffer, float * xGrid, float * yGrid, float * zGrid, vmml::vector<3, int> realDim, cudaStream_t stream)
{
	if (_grid)
		rayCasterGrid_CUDA(VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH, numRays, levelO, levelC, nLevel, _iso, cube, indexCube, cubeDim, cubeInc, _maxHeight, pixelBuffer, xGrid, yGrid, zGrid, VectorToInt3(realDim), _r, _g, _b, stream);
	else
		rayCaster_CUDA(VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH, numRays, levelO, levelC, nLevel, _iso, cube, indexCube, cubeDim, cubeInc, _maxHeight, VectorToInt3(_offset), pixelBuffer, _r, _g, _b, stream);
}

void rayCaster::renderCubes(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, visibleCube_t * cube,int * indexCube, int3 cubeDim, int3 cubeInc, float * pixelBuffer, float * xGrid, float * yGrid, float * zGrid, vmml::vector<3, int> realDim, cudaStream_t stream)
{
	if (_grid)
		rayCasterGrid_Cubes_CUDA(VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH, numRays, levelO, levelC, nLevel, _iso, cube, indexCube, cubeDim, cubeInc, _maxHeight, pixelBuffer, xGrid, yGrid, zGrid, VectorToInt3(realDim), _r, _g, _b, stream);
	else
		rayCaster_Cubes_CUDA(VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH, numRays, levelO, levelC, nLevel, _iso, cube, indexCube, cubeDim, cubeInc, _maxHeight, VectorToInt3(_offset), pixelBuffer, _r, _g, _b, stream);
}


}
