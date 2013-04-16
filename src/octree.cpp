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
	cudaFree(_GstackActual);
	cudaFree(_GstackIndex);
	cudaFree(_GstackLevel);
}

void Octree::setOctree(OctreeContainer * oc, int maxRays)
{
    _maxRays = maxRays;
    _nLevels = oc->getnLevels();
    _maxLevel = oc->getMaxLevel();
    _currentLevel = _maxLevel;

	_octree = oc->getOctree();
	_sizes = oc->getSizes();

	// Create octree State
	std::cerr<<"Allocating memory octree state stackIndex "<<_maxRays*STACK_DIM*sizeof(index_node_t)/1024.0f/1024.0f<<" MB: ";
	if (cudaSuccess != cudaMalloc(&_GstackIndex, _maxRays*STACK_DIM*sizeof(index_node_t)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	std::cerr<<"Allocating memory octree state stackActual "<<_maxRays*sizeof(int)/1024.0f/1024.0f<<" MB: ";
	if (cudaSuccess != cudaMalloc(&_GstackActual, _maxRays*sizeof(int)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	std::cerr<<"Allocating memory octree state stackLevel "<<_maxRays*STACK_DIM*sizeof(int)/1024.0f/1024.0f<<" MB: ";
	if (cudaSuccess != cudaMalloc(&_GstackLevel, maxRays*STACK_DIM*sizeof(int)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;
}


void Octree::resizeViewport(int width, int height)
{
	cudaFree(_GstackActual);
	cudaFree(_GstackIndex);
	cudaFree(_GstackLevel);

	_maxRays = width * height;

	// Create octree State
	std::cerr<<"Allocating memory octree state stackIndex "<<_maxRays*STACK_DIM*sizeof(index_node_t)/1024.0f/1024.0f<<" MB: ";
	if (cudaSuccess != cudaMalloc(&_GstackIndex, _maxRays*STACK_DIM*sizeof(index_node_t)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	std::cerr<<"Allocating memory octree state stackActual "<<_maxRays*sizeof(int)/1024.0f/1024.0f<<" MB: ";
	if (cudaSuccess != cudaMalloc(&_GstackActual, _maxRays*sizeof(int)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	std::cerr<<"Allocating memory octree state stackLevel "<<_maxRays*STACK_DIM*sizeof(int)/1024.0f/1024.0f<<" MB: ";
	if (cudaSuccess != cudaMalloc(&_GstackLevel, _maxRays*STACK_DIM*sizeof(int)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;
}


void Octree::resetState(cudaStream_t stream)
{
	resetStateOctree(stream, _GstackActual, _GstackIndex, _GstackLevel, _maxRays);	
}

void Octree::getBoxIntersected(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, cudaStream_t stream)
{

	getBoxIntersectedOctree(_octree, _sizes, _nLevels, origin, LB, up, right, w, h, pvpW, pvpH, _currentLevel, _maxRays, _GstackActual, _GstackIndex, _GstackLevel, visibleGPU, visibleCPU, stream);
}

}
