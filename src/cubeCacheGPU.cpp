/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <cubeCacheGPU.h>
#include <mortonCodeUtil_CPU.h>

namespace eqMivt
{
	typedef struct
	{
		cubeCacheCPU *        cpuCache;
		index_node_t            idCube;
	} callback_struct_t;

	void unLockCPUCube(cudaStream_t stream, cudaError_t status, void *data)
	{
		callback_struct_t * packet = (callback_struct_t*) data;

		packet->cpuCache->pop_cube(packet->idCube);

		delete packet;
	}



cubeCacheGPU::cubeCacheGPU()
{
	_cpuCache = 0;
	_cubeDim.set(0,0,0);
	_cubeInc.set(0,0,0);
	_realcubeDim.set(0,0,0);
	_offsetCube = 0;
	_levelCube = 0;
	_nLevels = 0;

	_queuePositions = 0;

	_maxElements = 0;
	_cacheData = 0;
}

cubeCacheGPU::~cubeCacheGPU()
{
	if (_queuePositions != 0)
		delete _queuePositions;
	if (_cacheData!=0)
	{
		int d = 40;
		cudaGetDevice(&d);
		if (d != _device)
			cudaSetDevice(_device);
		cudaFree(_cacheData);
		if (d != _device)
			cudaSetDevice(d);
	}
}

bool cubeCacheGPU::init(cubeCacheCPU * cpuCache, uint32_t device)
{
	if (device < 0)
		return false;
	_device = device;
	_cpuCache= cpuCache;
	_nLevels = _cpuCache->getnLevels(); 

}

bool cubeCacheGPU::reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int numElements)
{
	if (_cpuCache->getLevelCube() > levelCube)
	{
		LBERROR << "Cache GPU: level cube in gpu cache has to be >= level cube in cache CPU"<<std::endl;
		return false;
	}

	if (levelCube == _levelCube)
		return true;

	// cube size
	_cubeDim 	= cubeDim;
	_cubeInc.set(cubeInc,cubeInc,cubeInc);
	_realcubeDim	= cubeDim + 2 * cubeInc;
	_levelCube	= levelCube;
	_offsetCube	= (_cubeDim.x()+2*_cubeInc.x())*(_cubeDim.y()+2*_cubeInc.y())*(_cubeDim.z()+2*_cubeInc.z());

	if (numElements == 0)
	{
		size_t total = 0;
		size_t free = 0;

		if (cudaSuccess != cudaMemGetInfo(&free, &total))
		{
			LBERROR<<"Cache GPU: Error getting memory info"<<std::endl;
			return false;
		}

		float freeS = (8.0f*free)/10.0f; // Get 80% of free memory
		_maxElements = freeS / (float)(_offsetCube*sizeof(float));
		LBINFO << total/1024/1024 <<" "<<free /1024/1024<< " "<<freeS/1024/1024<<" " <<_maxElements<<std::endl;
	}
	else
	{
		_maxElements	= numElements;
	}

	_indexStored.clear();

	if (_queuePositions != 0)
		delete _queuePositions;
	_queuePositions  = new LinkedList(_maxElements);

	if (_cacheData!=0)
		cudaFree(_cacheData);
	// Allocating memory                                                                            
	LBINFO<<"Creating cache in GPU: "<< _maxElements*_offsetCube*sizeof(float)/1024/1024<<" MB"<<std::endl;
	if (cudaSuccess != cudaMalloc((void**)&_cacheData, _maxElements*_offsetCube*sizeof(float)))
	{                                                                                               
		LBERROR<<"Cache GPU: Error creating gpu cache"<<std::endl;
		return false;                                                                                  
	}       

	return true;
}

float * cubeCacheGPU::push_cube(index_node_t idCube, cudaStream_t stream)
{
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;

	_lock.set();

	float * cube = 0;

	// Find the cube in the GPU cache
	it = _indexStored.find(idCube);
	if ( it != _indexStored.end() ) // If exist
	{
		NodeLinkedList * node = it->second;

		unsigned pos    = node->element;
		cube    = _cacheData + pos*_offsetCube;

		_queuePositions->moveToLastPosition(node);
		_queuePositions->addReference(node);
	}
	else // If not exists
	{
		index_node_t     removedCube = (index_node_t)0;
		NodeLinkedList * node = _queuePositions->getFirstFreePosition(idCube, &removedCube);

		if (node != NULL)
		{
			index_node_t idCubeCPU = idCube >> 3*(_levelCube - _cpuCache->getLevelCube()); 
			// Search in cpu cache and check as locked
			float * pCube = _cpuCache->push_cube(idCubeCPU);

			// search on CPU cache
			if (pCube != NULL)
			{
				_indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
				if (removedCube!= (index_node_t)0)
					_indexStored.erase(_indexStored.find(removedCube));

				_queuePositions->moveToLastPosition(node);
				_queuePositions->addReference(node);

				unsigned pos   = node->element;
				cube    = _cacheData + pos*_offsetCube;

				if (idCube == idCubeCPU)
				{
					if (cudaSuccess != cudaMemcpyAsync((void*) cube, (void*) pCube, _offsetCube*sizeof(float), cudaMemcpyHostToDevice, stream))
					{
						LBERROR<<"Cache GPU: error copying to a device "<<cube<<" "<<pCube<<" "<<_offsetCube<<std::endl;
						throw;
					}
				}
				else
				{
					vmml::vector<3, int> realDimCPU = _cpuCache->getRealCubeDim();	
					size_t spitch = _realcubeDim.y()*_realcubeDim.z()*sizeof(float);
					size_t width = _realcubeDim.y()*_realcubeDim.z();
					size_t height = _realcubeDim.x();
					size_t dpitch = realDimCPU.y()*realDimCPU.z()*sizeof(float);

					vmml::vector<3, int> coord = getMinBoxIndex2(idCube, _levelCube, _nLevels);
					pCube += (coord.z() + coord.y()*realDimCPU.z() + coord.x()*realDimCPU.z()*realDimCPU.y()); 

					if (cudaSuccess != cudaMemcpy2DAsync((void*)cube, dpitch, (const void*)pCube, spitch, width, height, cudaMemcpyHostToDevice, stream))
					{
						LBERROR<<"Cache GPU: error copying to a device "<<cube<<" "<<pCube<<" "<<_offsetCube<<std::endl;
						throw;
					}

				}

				// Unlock the cube on cpu cache
				callback_struct_t * callBackData = new callback_struct_t;
				callBackData->cpuCache = _cpuCache;
				callBackData->idCube = idCubeCPU;

				if ( cudaSuccess != cudaStreamAddCallback(stream, unLockCPUCube, (void*)callBackData, 0))
				{
					LBERROR<<"Error making cudaCallback"<<std::endl;
					throw;
				}

			}

		}
		else // there is no free slot
		{
			return 0;
		}
	}

	_lock.unset();

	return cube;
}

void  cubeCacheGPU::pop_cube(index_node_t idCube)
{
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;

	_lock.set();

	// Find the cube in the GPU cache
	it = _indexStored.find(idCube);
	if ( it != _indexStored.end() ) // If exist remove reference
	{
		NodeLinkedList * node = it->second;
		_queuePositions->removeReference(node);
	}
	else
	{
		LBERROR<<"Cache GPU: is unistable"<<std::endl;
		throw;
	}

	_lock.unset();
}
}
