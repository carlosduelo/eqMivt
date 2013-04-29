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
    _init = false;

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
	if (_height < height || _width < width)
	{
		_height = height;
		_width = width;

		// Resize VisibleCubes
		_DestroyVisibleCubes();
		_CreateVisibleCubes();
	}

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

bool Render::checkCudaResources()
{
    return _init;
}

void Render::setCudaResources(OctreeContainer * oc, cubeCache * cc, int id)
{
    _octree.setOctree(oc, _height*_width);
	_cache = cc;
	_id  = id;
	_raycaster.setIsosurface(oc->getIsosurface());
	_init = true;
}

void Render::frameDraw(eq::Vector4f origin, eq::Vector4f LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, eq::Vector2f jitter)
{
	bool notEnd		= true;
	int numPixels	= pvpW*pvpH;
	int	iterations = 0;

	// Reset VisibleCubes 
	cudaMemsetAsync((void*)_visibleCubesGPU, 0, numPixels*sizeof(visibleCube_t), _stream);
	
	float * pixelBuffer;
    if (cudaSuccess != cudaGraphicsMapResources(1, &_cuda_pbo_resource, _stream))
    {
    	std::cerr<<"Error cudaGraphicsMapResources"<<std::endl;
    }

    size_t num_bytes;
    if (cudaSuccess != cudaGraphicsResourceGetMappedPointer((void **)&pixelBuffer, &num_bytes, _cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsResourceGetMappedPointer"<<std::endl;
    }
    std::cout<<"CUDA MAPPED "<<num_bytes<<std::endl;


	while(notEnd)
	{
		_octree.getBoxIntersected(origin, LB, up, right, w, h, pvpW, pvpH, jitter, _visibleCubesGPU, _visibleCubesCPU, _stream);

		cudaStreamSynchronize(_stream);

		#if 0
		int numP = 0;
		int nocached = 0;
		int painted = 0;
		int cube = 0;
		int cached = 0;
		int nocube = 0;
		for(int i=0; i<numPixels; i++)
			if (_visibleCubesCPU[i].state == PAINTED)
				numP++;
			else if (_visibleCubesCPU[i].state == NOCACHED)
				nocached++;
			else if (_visibleCubesCPU[i].state == CACHED)
				cached++;
			else if (_visibleCubesCPU[i].state == NOCUBE)
				nocube++;
			else if (_visibleCubesCPU[i].state == CUBE)
				cube++;

		std::cout<<"Painted "<<numP<<" NOCACHED "<<nocached<<" cached "<<cached<<" nocube "<<nocube<<" cube "<<cube<<std::endl;
		#endif

		if(!_cache->push(_visibleCubesCPU, numPixels, _octree.getOctreeLevel(), _id, _stream))
		{
			break;
		}

		cudaMemcpyAsync((void*) _visibleCubesGPU, (const void*) _visibleCubesCPU, (_height*_width)*sizeof(visibleCube_t), cudaMemcpyHostToDevice, _stream);

		vmml::vector<3, int> cDim = _cache->getCubeDim();
		vmml::vector<3, int> cInc = _cache->getCubeInc();

		_raycaster.render(origin, LB, up, right, w, h, pvpW, pvpH, jitter, (_height*_width), _octree.getOctreeLevel(), _cache->getCacheLevel(), _octree.getnLevels(), _visibleCubesGPU,  make_int3(cDim.x(), cDim.y(), cDim.z()), make_int3(cInc.x(), cInc.y(), cInc.z()), pixelBuffer, _stream);

		_cache->pop(_visibleCubesCPU, numPixels, _octree.getOctreeLevel(), _id, _stream);
		
		//std::cout<<iterations<<std::endl;
		iterations++;
	}

    if (cudaSuccess != cudaGraphicsUnmapResources(1, &_cuda_pbo_resource, _stream))
    {
    	std::cerr<<"Error cudaGraphicsUnmapResources"<<std::endl;
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
