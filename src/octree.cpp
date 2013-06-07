#include "octree.h"

#include "octree_CUDA.h"

#include <iostream>
#include <fstream>

#define STACK_DIM 32

#define VectorToFloat3(v) make_float3((v).x(), (v).y(), (v).z())

namespace eqMivt
{

Octree::Octree()
{
	_isosurface = -1.0f;
	_realDim.set(0,0,0);
	_dimension = 0;
	_nLevels = 0;
	_maxLevel = 0;

	_memoryOctree = 0;
	_octree = 0;
	_sizes = 0;
	_currentLevel = 0;
	_device = 40;
}

Octree::~Octree()
{
	if (!Destroy_Octree(_device, _octree, _memoryOctree, _sizes))
	{
		std::cerr<<"Error deleting a octree"<<std::endl;
	}
}

void Octree::setGeneralValues(vmml::vector<3, int> realDim, int dimension, int nLevels, int maxLevel, uint32_t device)
{
	_device = device;
	_realDim = realDim;
	_dimension = dimension;
	_nLevels = nLevels;
	_maxLevel = maxLevel;
}

bool Octree::setCurrentOctree(int currentLevel, float isosurface,  int maxHeight, index_node_t ** octree, int * sizes)
{
	_currentLevel = currentLevel;
	if (_isosurface != isosurface)
	{
		_maxHeight = maxHeight;
		_isosurface = isosurface;
		return Create_Octree(octree, sizes, _maxLevel, &_octree, &_memoryOctree, &_sizes);
	}
	return true;
}

void Octree::getBoxIntersected(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, int numRays, int * indexVisibleCubesGPU, int * indexVisibleCubesCPU, cudaStream_t stream)
{

	getBoxIntersectedOctree(_octree, _sizes, _nLevels, VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH,  _currentLevel, numRays, visibleGPU, visibleCPU, indexVisibleCubesGPU, indexVisibleCubesCPU, stream);
}

}
