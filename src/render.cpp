/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "render.h"
#include <iostream>

#include "rayGenerator.h"

namespace eqMivt
{

Render::Render()
{
    _init = false;

    _height = 0;
    _width = 0;

    _visibleCubesGPU = 0;
    _visibleCubesCPU = 0;

	_rays = 0;

    _cuda_pbo_resource = 0;

    if (cudaSuccess != cudaStreamCreate(&_stream))
    {
	    std::cerr<<"Error cudaStreamCreate"<<std::endl;
    }
}

Render::~Render()
{

	if (_rays != 0)
		cudaFree(_rays);

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


	// Resize Rays
    if (cudaSuccess != (cudaMalloc(&_rays, (3*_height*_width)*sizeof(float))))
    {
    	std::cerr<< "Render: error allocating rays in the gpu\n";
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

void Render::frameDraw(eq::Vector4f origin, eq::Vector4f LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH)
{
	// Reset VisibleCubes 
	cudaMemsetAsync((void*)_visibleCubesGPU, 0, (_height*_width)*sizeof(visibleCube_t), _stream);
	
	//Generate rays
	generateRays_CUDA(_rays, make_float3(origin.x(),origin.y(),origin.z()), make_float3(LB.x(),LB.y(),LB.z()), make_float3(up.x(),up.y(),up.z()), make_float3(right.x(),right.y(),right.z()), w, h, pvpW, pvpH, _stream);

	float * pixelBuffer;
    if (cudaSuccess != cudaGraphicsMapResources(1, &_cuda_pbo_resource, 0))
    {
    	std::cerr<<"Error cudaGraphicsMapResources"<<std::endl;
    }
    size_t num_bytes;
    if (cudaSuccess != cudaGraphicsResourceGetMappedPointer((void **)&pixelBuffer, &num_bytes, _cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsResourceGetMappedPointer"<<std::endl;
    }
    std::cout<<"CUDA MAPPED "<<num_bytes<<std::endl;

	bool notEnd		= true;
	int numPixels	= _height*_width;
	int	iterations = 0;

	while(notEnd)
	{
		_octree.getBoxIntersected(make_float3(origin.x(),origin.y(),origin.z()), _rays, pvpW, pvpH, _visibleCubesGPU, _visibleCubesCPU, _stream);

		cudaStreamSynchronize(_stream);

		int numP = 0;
		#if 0
		int nocached = 0;
		int painted = 0;
		int cube = 0;
		int cached = 0;
		int nocube = 0;
		#endif
		for(int i=0; i<numPixels; i++)
			if (_visibleCubesCPU[i].state == PAINTED)
				numP++;
		#if 0
			else if (_visibleCubesCPU[i].state == NOCACHED)
				nocached++;
			else if (_visibleCubesCPU[i].state == CACHED)
				cached++;
			else if (_visibleCubesCPU[i].state == NOCUBE)
				nocube++;
			else if (_visibleCubesCPU[i].state == CUBE)
				cube++;
		#endif

		if (numP == numPixels)
		{
			notEnd = false;
			break;
		}

		//std::cout<<"Painted "<<numP<<" NOCACHED "<<nocached<<" cached "<<cached<<" nocube "<<nocube<<" cube "<<cube<<std::endl;

		_cache->push(_visibleCubesCPU, (_height*_width), _octree.getOctreeLevel(), _id, _stream);

		cudaMemcpyAsync((void*) _visibleCubesGPU, (const void*) _visibleCubesCPU, (_height*_width)*sizeof(visibleCube_t), cudaMemcpyHostToDevice, _stream);

		vmml::vector<3, int> cDim = _cache->getCubeDim();
		vmml::vector<3, int> cInc = _cache->getCubeInc();

		_raycaster.render(make_float3(origin.x(),origin.y(),origin.z()), _rays, (_height*_width), _octree.getOctreeLevel(), _cache->getCacheLevel(), _octree.getnLevels(), _visibleCubesGPU,  make_int3(cDim.x(), cDim.y(), cDim.z()), make_int3(cInc.x(), cInc.y(), cInc.z()), pixelBuffer, _stream);

		_cache->pop(_visibleCubesCPU, (_height*_width), _octree.getOctreeLevel(), _id, _stream);
		
		std::cout<<iterations<<std::endl;
		iterations++;
	}



    if (cudaSuccess != cudaGraphicsUnmapResources(1, &_cuda_pbo_resource, 0))
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
