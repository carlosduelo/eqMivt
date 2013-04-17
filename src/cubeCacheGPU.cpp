/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <cubeCacheGPU.h>

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
}

cubeCacheGPU::~cubeCacheGPU()
{
	delete queuePositions;
	cudaFree(cacheData);
}

bool cubeCacheGPU::init(cubeCacheCPU * p_cpuCache, int p_maxElements)
{
	cpuCache 	= p_cpuCache;
	maxElements	= p_maxElements;

	// cube size
	cubeDim         = cpuCache->getCubeDim();
	cubeInc         = cpuCache->getCubeInc(); 
	realcubeDim     = cubeDim + 2 * cubeInc.x();
	levelCube       = cpuCache->getLevelCube();
	nLevels         = cpuCache->getnLevels();
	offsetCube      = (cubeDim.x()+2*cubeInc.x())*(cubeDim.y()+2*cubeInc.y())*(cubeDim.z()+2*cubeInc.z());

	// Creating caches
	queuePositions  = new LinkedList(maxElements);

	// Allocating memory                                                                            
	LBINFO<<"Creating cache in GPU: "<< maxElements*offsetCube*sizeof(float)/1024/1024<<" MB"<<std::endl;
	if (cudaSuccess != cudaMalloc((void**)&cacheData, maxElements*offsetCube*sizeof(float)))
	{                                                                                               
		LBERROR<<"LRUCache: Error creating gpu cache"<<std::endl;
		return false;                                                                                  
	}       

	return true;
}

float * cubeCacheGPU::push_cube(index_node_t idCube, cudaStream_t stream)
{
#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif

	lock.set();

	float * cube = 0;

	// Find the cube in the GPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist
	{
		NodeLinkedList * node = it->second;

		unsigned pos    = node->element;
		cube    = cacheData + pos*offsetCube;

		queuePositions->moveToLastPosition(node);
		queuePositions->addReference(node);
	}
	else // If not exists
	{
		index_node_t     removedCube = (index_node_t)0;
		NodeLinkedList * node = queuePositions->getFirstFreePosition(idCube, &removedCube);

		if (node != NULL)
		{
			// Search in cpu cache and check as locked
			float * pCube = cpuCache->push_cube(idCube);

			// search on CPU cache
			if (pCube != NULL)
			{
				indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
				if (removedCube!= (index_node_t)0)
					indexStored.erase(indexStored.find(removedCube));

				queuePositions->moveToLastPosition(node);
				queuePositions->addReference(node);

				unsigned pos   = node->element;
				cube    = cacheData + pos*offsetCube;

				if (cudaSuccess != cudaMemcpyAsync((void*) cube, (void*) pCube, offsetCube*sizeof(float), cudaMemcpyHostToDevice, stream))
				{
					std::cerr<<"Cache GPU_CPU_File: error copying to a device"<<std::endl;
				}

				// Unlock the cube on cpu cache

				callback_struct_t * callBackData = new callback_struct_t;
				callBackData->cpuCache = cpuCache;
				callBackData->idCube = idCube;

				if ( cudaSuccess != cudaStreamAddCallback(stream, unLockCPUCube, (void*)callBackData, 0))
				{
					std::cerr<<"Error making cudaCallback"<<std::endl;
					throw;
				}

			}

		}
		else // there is no free slot
		{
		}
	}

	lock.unset();

	return cube;
}

void  cubeCacheGPU::pop_cube(index_node_t idCube)
{
#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif

	lock.set();

	// Find the cube in the GPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist remove reference
	{
		NodeLinkedList * node = it->second;
		queuePositions->removeReference(node);
	}
	else
	{
		std::cerr<<"Cache is unistable"<<std::endl;
		throw;
	}

	lock.unset();
}
}
