#include "octree.h"

#include "octree_CUDA.h"

#include <iostream>
#include <fstream>

#define STACK_DIM 32

#define VectorToFloat3(v) make_float3((v).x(), (v).y(), (v).z())
#define VectorToInt3(v) make_int3((v).x(), (v).y(), (v).z())

namespace eqMivt
{

Octree::Octree()
{
	_isosurface = -1.0f;
	_realDim.set(0,0,0);
	_dimension = 0;
	_nLevels = 0;
	_maxLevel = 0;

	_xGrid = 0;
	_yGrid = 0;
	_zGrid = 0;
	_memoryOctree = 0;
	_octree = 0;
	_sizes = 0;
	_currentLevel = 0;
	_device = 40;
}

Octree::~Octree()
{
	if (!Destroy_Octree(_device, _octree, _memoryOctree, _sizes, _xGrid, _yGrid, _zGrid))
	{
		std::cerr<<"Error deleting a octree"<<std::endl;
	}
}

void Octree::setGeneralValues(uint32_t device)
{
	_device = device;
}

bool Octree::setCurrentOctree(vmml::vector<3, int> realDim, int dimension, int nLevels, int maxLevel, int currentLevel, float isosurface,  int maxHeight, index_node_t ** octree, int * sizes, double * xGrid, double * yGrid, double * zGrid, vmml::vector<3, int> realVolDim, int lastLevel)
{
	_currentLevel = currentLevel;
	if (_isosurface != isosurface || _dimension != dimension ||
		_nLevels != nLevels || _maxLevel != maxLevel)
	{
		_realDim = realDim;
		_dimension = dimension;
		_nLevels = nLevels;
		_maxLevel = maxLevel;
		_maxHeight = maxHeight;
		_isosurface = isosurface;
		return Create_Octree(octree, sizes, _maxLevel, &_octree, &_memoryOctree, &_sizes, lastLevel, VectorToInt3(realVolDim), VectorToInt3(realDim), &_xGrid, &_yGrid, &_zGrid, xGrid, yGrid, zGrid);
	}
	return true;
}

void Octree::getBoxIntersected(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, int numRays, int * indexVisibleCubesGPU, int * indexVisibleCubesCPU, cudaStream_t stream)
{

	getBoxIntersectedOctree(_octree, _sizes, _nLevels, VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH,  _currentLevel, numRays, visibleGPU, visibleCPU, indexVisibleCubesGPU, indexVisibleCubesCPU, stream);
}

}
