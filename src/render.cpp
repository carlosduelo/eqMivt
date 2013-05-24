/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "render.h"
#include <iostream>

#define MAX_ITERATIONS 250

namespace eqMivt
{

Render::Render()
{
    _init = false;

    _height = 0;
    _width = 0;

    _visibleCubesGPU = 0;
    _visibleCubesCPU = 0;
	_indexVisibleCubesGPU = 0;
	_indexVisibleCubesCPU = 0;

	_cuda_pbo_resource = 0;
	_stream = 0;

	_name = "no-name";
	_outputFile = 0;
	_statistics = false;
	_resizeTimes = 0;
	_resizeAccum = 0.0;
	_mapResourcesTimes = 0;
	_mapResourcesAccum = 0.0;
	_unmapResourcesTimes = 0;
	_unmapResourcesAccum = 0.0;
	_octreeTimes = 0;
	_octreeAccum = 0.0;
	_rayCastingTimes = 0;
	_rayCastingAccum = 0.0;
	_cachePushTimes = 0;
	_cachePushAccum = 0.0;
	_cachePopTimes = 0;
	_cachePopAccum = 0.0;
	_frameDrawTimes = 0;
	_frameDrawAccum = 0.0;
}

Render::~Render()
{

    // Destroy Visible cubes
    _DestroyVisibleCubes();

	if (cudaSuccess != cudaStreamSynchronize(_stream))
	{
		std::cerr<<"Error cudaStreamSynchronize"<<std::endl;
	}

    if (cudaSuccess != cudaGraphicsUnmapResources(1, &_cuda_pbo_resource, _stream))
    {
    	std::cerr<<"Error cudaGraphicsUnmapResources"<<std::endl;
    }

    // Destroy Stream
    if (cudaSuccess != cudaStreamDestroy(_stream))
    {
	    std::cerr<<"Error cudaStreamDestroy"<<std::endl;
    }

    if (_cuda_pbo_resource != 0 && cudaSuccess != cudaGraphicsUnregisterResource(_cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsUnregisterResource"<<std::endl;
    }

	if (_outputFile != 0)
		delete _outputFile;
}

void Render::resizeViewport(int width, int height, GLuint pbo)
{
	_resizeTimes++;
	_resizeClock.reset();

	if (_height != height || _width != width)
	{
		_height = height;
		_width = width;

		// Resize VisibleCubes
		_DestroyVisibleCubes();
		_CreateVisibleCubes();
	}

    if (_stream == 0)
	{
		int dev = -1;
		if (cudaSuccess == cudaGetDevice(&dev) && cudaSuccess == cudaGetDeviceProperties(&_cudaProp, dev))
			std::cout<<dev<<" "<<_cudaProp.name<<std::endl;
		if (cudaSuccess != cudaStreamCreate(&_stream))
		{
			std::cerr<<"Error cudaStreamCreate"<<std::endl;
		}
	}

	if (cudaSuccess != cudaStreamSynchronize(_stream))
	{
		std::cerr<<"Error cudaStreamSynchronize"<<std::endl;
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

	double time = _resizeClock.getTimed();
	_resizeAccum += time;
	if (_statistics)
		*_outputFile<<"Resize time "<<time/1000.0<<" seconds"<<std::endl;
}

bool Render::checkCudaResources()
{
    return _init;
}

void Render::setStatistics(bool stat)
{ 
	_statistics = stat;
	if (stat && _outputFile == 0)
	{
		_outputFile = new std::ofstream((_name+".log").c_str(), std::ofstream::trunc );
		*_outputFile<<"Render in graphic card: "<<_cudaProp.name<<std::endl;
	}
}

void Render::printCudaProperties()
{
	std::cout<<_cudaProp.name<<std::endl;
}

void Render::setCudaResources(OctreeContainer * oc, cubeCache * cc, int id, std::string name)
{
    _octree.setOctree(oc);
	_cache = cc;
	_id  = id;
	_raycaster.setIsosurface(oc->getIsosurface());
	_init = true;
	_name = name + "-Render";
}

void Render::frameDraw(eq::Vector4f origin, eq::Vector4f LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH)
{
	_frameDrawTimes++;
	_frameDrawClock.reset();


	int numPixels	= pvpW*pvpH;
	int	iterations = 0;

	_mapResourcesTimes++;
	_mapResourcesClock.reset();

	// Reset VisibleCubes 
	if (cudaSuccess != cudaMemsetAsync((void*)_visibleCubesGPU, 0, numPixels*sizeof(visibleCube_t), _stream))
	{
		std::cerr<<"Error initialize visible cubes"<<std::endl;
	}
	for(int i=0; i<numPixels; i++)
		_indexVisibleCubesCPU[i] = i;
	cudaMemcpyAsync((void*) _indexVisibleCubesGPU, (const void*) _indexVisibleCubesCPU, numPixels*sizeof(int), cudaMemcpyHostToDevice, _stream);
	
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

	if (cudaSuccess != cudaMemsetAsync((void*)pixelBuffer, (int)1.0f, num_bytes, _stream))
	{
		std::cerr<<"Error initialize visible cubes"<<std::endl;
	}

	if (_statistics)
		cudaStreamSynchronize(_stream);

	double time = _mapResourcesClock.getTimed();
	_mapResourcesAccum += time;
	if (_statistics)
		*_outputFile<<"Time to map cuda resources and initialization "<<time/1000.0 <<" seconds"<<std::endl;

	double timeO = 0.0;
	double timeCP = 0.0;
	double timeR = 0.0;
	double timeCp = 0.0;

	while(iterations < MAX_ITERATIONS)
	{
		_octreeTimes++;
		_octreeClock.reset();

		_octree.getBoxIntersected(origin, LB, up, right, w, h, pvpW, pvpH, _visibleCubesGPU, _visibleCubesCPU, numPixels, _indexVisibleCubesGPU, _indexVisibleCubesCPU, _stream);

		cudaMemcpyAsync((void*) _visibleCubesCPU, (const void*) _visibleCubesGPU, pvpW*pvpH*sizeof(visibleCube_t), cudaMemcpyDeviceToHost, _stream);

		if (cudaSuccess != cudaStreamSynchronize(_stream))
		{
			std::cerr<<"Error cudaStreamSynchronize waiting octree: "<<cudaGetErrorString(cudaGetLastError()) <<std::endl;
		}

		time = _octreeClock.getTimed();
		_octreeAccum += time;
		timeO += time;

		_cachePushTimes++;
		_cachePushClock.reset();

		if(!_cache->push(_visibleCubesCPU, _indexVisibleCubesCPU, &numPixels, _octree.getOctreeLevel(), _id, _stream))
		{
			break;
		}

		cudaMemcpyAsync((void*) _visibleCubesGPU, (const void*) _visibleCubesCPU, pvpW*pvpH*sizeof(visibleCube_t), cudaMemcpyHostToDevice, _stream);
		cudaMemcpyAsync((void*) _indexVisibleCubesGPU, (const void*) _indexVisibleCubesCPU, numPixels*sizeof(int), cudaMemcpyHostToDevice, _stream);

		if (_statistics)
		{
			if (cudaSuccess != cudaStreamSynchronize(_stream))
			{
				std::cerr<<"Error cudaStreamSynchronize waiting transfering visibleCubes to GPU: "<<cudaGetErrorString(cudaGetLastError()) <<std::endl;
			}

			*_outputFile<<"Time octree: "<<time/1000.0 <<" seconds"<<std::endl;

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

			*_outputFile<<"Painted "<<(numP*100.0)/(float)numPixels<<" NOCACHED "<<(nocached*100.0)/(float)numPixels<<" cached "<<(cached*100.0)/(float)numPixels<<" nocube "<<(nocube*100.0)/(float)numPixels<<" cube "<<(cube*100.0)/(float)numPixels<<std::endl;
		}

		time = _cachePushClock.getTimed();
		_cachePushAccum += time;
		timeCP += time;
		if (_statistics)
			*_outputFile<<"Time cache push: "<<time/1000.0 <<" seconds"<<std::endl;

		_rayCastingTimes++;
		_rayCastingClock.reset();

		vmml::vector<3, int> cDim = _cache->getCubeDim();
		vmml::vector<3, int> cInc = _cache->getCubeInc();

		_raycaster.render(origin, LB, up, right, w, h, pvpW, pvpH, numPixels, _octree.getOctreeLevel(), _cache->getCacheLevel(), _octree.getnLevels(), _visibleCubesGPU,  _indexVisibleCubesGPU, make_int3(cDim.x(), cDim.y(), cDim.z()), make_int3(cInc.x(), cInc.y(), cInc.z()), pixelBuffer, _stream);

		if (_statistics)
			if (cudaSuccess != cudaStreamSynchronize(_stream))
			{
				std::cerr<<"Error cudaStreamSynchronize waiting ray casting: "<<cudaGetErrorString(cudaGetLastError()) <<std::endl;
			}

		time = _rayCastingClock.getTimed();
		_rayCastingAccum += time;
		timeR += time;
		if (_statistics)
			*_outputFile<<"Time ray casting: "<<time/1000.0 <<" seconds"<<std::endl;

		_cachePopTimes++;
		_cachePopClock.reset();

		_cache->pop(_visibleCubesCPU, _indexVisibleCubesCPU, numPixels, _octree.getOctreeLevel(), _id, _stream);
		time = _cachePopClock.getTimed();
		_cachePopAccum += time;
		timeCp += time;
		if (_statistics)
			*_outputFile<<"Time cache pop: "<<time/1000.0 <<" seconds"<<std::endl;
		
		iterations++;
		//std::cout<<iterations<<std::endl;
	}

	_unmapResourcesTimes++;
	_unmapResourcesClock.reset();

    if (cudaSuccess != cudaGraphicsUnmapResources(1, &_cuda_pbo_resource, _stream))
    {
    	std::cerr<<"Error cudaGraphicsUnmapResources"<<std::endl;
    }

	time = _unmapResourcesClock.getTimed();
	_unmapResourcesAccum += time;
	if (_statistics)
		*_outputFile<<"Time to unmap cuda resources "<<time/1000.0 <<" seconds"<<std::endl;

	time = _frameDrawClock.getTimed();
	_frameDrawAccum += time;
	if (_statistics)
	{
		*_outputFile<<"Time to unmap cuda resources "<<time/1000.0 <<" seconds"<<std::endl;
		*_outputFile<<"Time to draw a frame "<<time/1000.0 <<" seconds"<<std::endl;
		*_outputFile<<"Iterations "<<iterations<<" Octree "<<(timeO*100.0)/time<<" cache push "<<(timeCP*100.0)/time<<" ray casting "<<(timeR*100.0)/time<<" cache pop "<<(timeCp*100.0)/time<<std::endl;
	}

}

void Render::_CreateVisibleCubes()
{
    if (cudaSuccess != (cudaMalloc(&_visibleCubesGPU, (_height*_width)*sizeof(visibleCube_t))))
    {
    	std::cerr<< "Render: error creating visible cubes in gpu"<<std::endl;
    }

	if (cudaSuccess != cudaHostAlloc((void**)&_visibleCubesCPU, _height*_width*sizeof(visibleCube_t),cudaHostAllocDefault))
	{
		std::cerr<<"Render: error creating visible cubes in cpu"<<std::endl;
	}

    if (cudaSuccess != (cudaMalloc(&_indexVisibleCubesGPU, (_height*_width)*sizeof(int))))
    {
    	std::cerr<< "Render: error creating visible cubes in gpu"<<std::endl;
    }

	if (cudaSuccess != cudaHostAlloc((void**)&_indexVisibleCubesCPU, _height*_width*sizeof(int),cudaHostAllocDefault))
	{
		std::cerr<<"Render: error creating visible cubes in cpu"<<std::endl;
	}
}

void Render::_DestroyVisibleCubes()
{
    if (_visibleCubesGPU != 0)
        cudaFree(_visibleCubesGPU);

    if (_visibleCubesCPU != 0)
		cudaFreeHost(_visibleCubesCPU);

    if (_indexVisibleCubesGPU!= 0)
        cudaFree(_indexVisibleCubesGPU);

    if (_indexVisibleCubesCPU!= 0)
		cudaFreeHost(_indexVisibleCubesCPU);
}

}
