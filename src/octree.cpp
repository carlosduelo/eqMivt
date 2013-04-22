#include "octree.h"

#include "octree_CUDA.h"

#include <iostream>
#include <fstream>

#define STACK_DIM 32

namespace eqMivt
{

Octree::Octree()
{
}

Octree::~Octree()
{
}

void Octree::setOctree(OctreeContainer * oc, int maxRays)
{
    _maxRays = maxRays;
    _nLevels = oc->getnLevels();
    _maxLevel = oc->getMaxLevel();
    _currentLevel = _maxLevel;

	_octree = oc->getOctree();
	_sizes = oc->getSizes();
}


void Octree::resizeViewport(int width, int height)
{
	_maxRays = width * height;
}

void Octree::getBoxIntersected(float3 origin, float * rays, int pvpW, int pvpH, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, cudaStream_t stream)
{
	
	getBoxIntersectedOctree(_octree, _sizes, _nLevels, origin, rays,  _currentLevel, pvpW*pvpH, visibleGPU, visibleCPU, stream);
}

}
