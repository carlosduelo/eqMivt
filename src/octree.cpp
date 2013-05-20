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
}

Octree::~Octree()
{
}

void Octree::setOctree(OctreeContainer * oc)
{
    _nLevels = oc->getnLevels();
    _maxLevel = oc->getMaxLevel();
    _currentLevel = _maxLevel;

	_octree = oc->getOctree();
	_sizes = oc->getSizes();
}


void Octree::getBoxIntersected(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, cudaStream_t stream)
{
	
	getBoxIntersectedOctree(_octree, _sizes, _nLevels, VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right), w, h, pvpW, pvpH,  _currentLevel, pvpW*pvpH, visibleGPU, visibleCPU, stream);
}

}
