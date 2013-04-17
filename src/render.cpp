/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "render.h"
#include <iostream>


namespace eqMivt
{

Render::Render()
{
    _initOctree = false;

    _height = 0;
    _width = 0;

    _visibleCubesGPU = 0;
    _visibleCubesCPU = 0;

    _cuda_pbo_resource = 0;

    if (cudaSuccess != cudaStreamCreate(&_stream))
    {
	    std::cerr<<"Error cudaStreamCreate"<<std::endl;
    }
}

Render::~Render()
{
    // Destroy Visible cubes
    _DestroyVisibleCubes();

    // Destroy Stream
    if (cudaSuccess != cudaStreamDestroy(_stream))
    {
	    std::cerr<<"Error cudaStreamDestroy"<<std::endl;
    }

    if (_cuda_pbo_resource != 0 && cudaSuccess != cudaGraphicsUnregisterResource(_cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsUnregisterResource"<<std::endl;
    }
}

void Render::resizeViewport(int width, int height, GLuint pbo)
{
    _height = height;
    _width = width;

    // Resize VisibleCubes
    _DestroyVisibleCubes();
    _CreateVisibleCubes();

    // Resize pbo
    if (_cuda_pbo_resource != 0 && cudaSuccess != cudaGraphicsUnregisterResource(_cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsUnregisterResource"<<std::endl;
    }
    if (cudaSuccess != cudaGraphicsGLRegisterBuffer(&_cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard))
    {
    	std::cerr<<"Error cudaGraphicsGLRegisterBuffer"<<std::endl;
    }


	_octree.resizeViewport(width, height);
}

bool Render::checkCudaResources()
{
    return _initOctree;
}

void Render::setCudaResources(OctreeContainer * oc)
{
    _octree.setOctree(oc, _height*_width);
	_initOctree = true;
}

void Render::frameDraw(eq::Vector4f origin, eq::Vector4f LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH)
{
	_octree.resetState(_stream);
	//_octree.getBoxIntersected(make_float3(origin.x(),origin.y(),origin.z()), make_float3(LB.x(),LB.y(),LB.z()), make_float3(up.x(),up.y(),up.z()), make_float3(right.x(),right.y(),right.z()), w, h, pvpW, pvpH, _visibleCubesGPU, _visibleCubesCPU, _stream);
}

void Render::_CreateVisibleCubes()
{
    if (cudaSuccess != (cudaMalloc(&_visibleCubesGPU, (_height*_width)*sizeof(visibleCube_t))))
    {
    	std::cerr<< "Octree: error allocating octree in the gpu\n";
    }

    _visibleCubesCPU = new visibleCube_t[_height*_width];
}

void Render::_DestroyVisibleCubes()
{
    if (_visibleCubesGPU != 0)
        cudaFree(_visibleCubesGPU);

    if (_visibleCubesCPU != 0)
    	delete[] _visibleCubesCPU;
}

}
