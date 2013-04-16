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
