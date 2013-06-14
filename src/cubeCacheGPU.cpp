/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <cubeCacheGPU.h>
#include <mortonCodeUtil_CPU.h>

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

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
	_minIndex = 0;
	_maxIndex = 0;
	_memorySize = -1.0f;

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

	return true;
}

bool cubeCacheGPU::reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int numElements)
{
	if (_cpuCache->getLevelCube() > levelCube)
	{
		LBERROR << "Cache GPU: level cube in gpu cache has to be >= level cube in cache CPU"<<std::endl;
		return false;
	}

	if (levelCube == _levelCube && _nLevels == _cpuCache->getnLevels())
		return true;

	// cube size
	_nLevels = _cpuCache->getnLevels(); 
	_cubeDim 	= cubeDim;
	_cubeInc.set(cubeInc,cubeInc,cubeInc);
	_realcubeDim	= cubeDim + 2 * cubeInc;
	_levelCube	= levelCube;
	_offsetCube	= _realcubeDim.x() * _realcubeDim.y() * _realcubeDim.z();
	_minIndex = coordinateToIndex(vmml::vector<3, int>(0,0,0), _levelCube, _nLevels); 
	int d = exp2(_nLevels);
	_maxIndex = coordinateToIndex(vmml::vector<3, int>(d-1,d-1,d-1), _levelCube, _nLevels);

	if (_memorySize < 0.0f)
	{
		size_t total = 0;
		size_t free = 0;

		if (cudaSuccess != cudaMemGetInfo(&free, &total))
		{
			LBERROR<<"Cache GPU: Error getting memory info"<<std::endl;
			return false;
		}

		_memorySize = (8.0f*free)/10.0f; // Get 80% of free memory
	}

	if (numElements == 0)
	{
		float cd = _offsetCube;
		cd *= sizeof(float);
		_maxElements = _memorySize/ cd;
		//LBINFO << total/1024/1024 <<" "<<free /1024/1024<< " "<<freeS/1024/1024<<" " <<_maxElements<<std::endl;
		if (_maxElements == 0)
		{
			LBERROR<<"Cache GPU: Memory aviable is not enough "<<_memorySize/1024/1024<<" MB"<<std::endl;
			return false;
		}
	}
	else
	{
		_maxElements	= numElements;
		if (_maxElements*_offsetCube*sizeof(float) > _memorySize)
		{
			LBERROR<<"Cache GPU: max elements in cache gpu are to big"<<std::endl;
			return false;
		}
	}

	_indexStored.clear();

	if (_queuePositions != 0)
		delete _queuePositions;
	_queuePositions  = new LinkedList(_maxElements);

	if (_cacheData == 0)
	{
//		cudaFree(_cacheData);
		// Allocating memory                                                                            
		LBINFO<<"Creating cache in GPU: "<< _memorySize/1024/1024<<" MB"<<std::endl;
		if (cudaSuccess != cudaMalloc((void**)&_cacheData, _memorySize))
		{                                                                                               
			LBERROR<<"Cache GPU: Error creating gpu cache"<<std::endl;
			return false;                                                                                  
		}
	}

	return true;
}

float * cubeCacheGPU::push_cube(index_node_t idCube, cudaStream_t stream)
{
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
	if (idCube < _minIndex || idCube > _maxIndex)
	{
		LBERROR<<"Cache GPU: Trying to push a worng index cube "<<idCube<<std::endl;
		return 0;
	}

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
					#if 0
					vmml::vector<3, int> coord = getMinBoxIndex2(idCube, _levelCube, _nLevels);
					vmml::vector<3, int> coordC = getMinBoxIndex2(idCubeCPU, _cpuCache->getLevelCube(), _nLevels) - _cpuCache->getCubeInc();
					coord -= coordC;
					vmml::vector<3, int> realDimCPU = _cpuCache->getRealCubeDim();

					cudaMemcpy3DParms myParms = {0};
					myParms.srcPos = make_cudaPos(coord.x(), coord.y(), coord.z());
					myParms.srcPtr = make_cudaPitchedPtr((void*)pCube, realDimCPU.z()*sizeof(float), realDimCPU.x(), realDimCPU.y()); 
					myParms.dstPos = make_cudaPos(0,0,0);
					myParms.dstPtr = make_cudaPitchedPtr((void*)cube, _realcubeDim.z()*sizeof(float), _realcubeDim.x(), _realcubeDim.y()); 
					myParms.extent = make_cudaExtent(_realcubeDim.x()*sizeof(float),_realcubeDim.y()*sizeof(float),_realcubeDim.z()*sizeof(float));
					myParms.kind = cudaMemcpyHostToDevice;

					if (cudaSuccess != cudaMemcpy3DAsync(&myParms, stream))
					{
						LBERROR<<"Cache GPU: error copying to a device "<<cube<<" "<<pCube<<" "<<_offsetCube<<std::endl;
						throw;
					}
					#endif
					// Not implemented
					std::cerr<<"NOT IMPLEMENTED"<<std::endl;
					throw;

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
