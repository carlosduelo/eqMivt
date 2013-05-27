/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <cubeCacheCPU.h>

#include <cuda_runtime.h>

namespace eqMivt
{
bool cubeCacheCPU::init(std::string type_file, std::vector<std::string> file_params, int p_maxElements, vmml::vector<3, int> p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels)
{
	// cube size
	cubeDim 	= p_cubeDim;
	cubeInc.set(p_cubeInc,p_cubeInc,p_cubeInc);
	realcubeDim	= p_cubeDim + 2 * p_cubeInc;
	levelCube	= p_levelCube;
	nLevels		= p_nLevels;
	offsetCube	= (cubeDim.x()+2*cubeInc.x())*(cubeDim.y()+2*cubeInc.y())*(cubeDim.z()+2*cubeInc.z());

	// Creating caches
	maxElements	= p_maxElements;
	queuePositions	= new LinkedList(maxElements);

	// OpenFile
	fileManager = eqMivt::CreateFileManage(type_file, file_params, levelCube, nLevels, cubeDim, cubeInc);
	if (fileManager == 0)
	{
		LBERROR<<"Cube Cache CPU: error initialization file"<<std::endl;
		return false;
	}

	// Allocating memory
	std::cerr<<"Creating cache in CPU: "<< maxElements*offsetCube*sizeof(float)/1024.0f/1024.0f<<" MB: "<<std::endl;
	if (cudaSuccess != cudaHostAlloc((void**)&cacheData, maxElements*offsetCube*sizeof(float),cudaHostAllocDefault))
	{
		LBERROR<<"Cube Cache CPU: Error creating cpu cache"<<std::endl;
		return false;
	}

	return true;
}

cubeCacheCPU::~cubeCacheCPU()
{
	delete fileManager;
	delete queuePositions;
	cudaFreeHost(cacheData);
}

float * cubeCacheCPU::push_cube(index_node_t idCube)
{
#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif
	lock.set();
	
	// Find the cube in the CPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist
	{
		NodeLinkedList * node = it->second;

		queuePositions->moveToLastPosition(node);
		queuePositions->addReference(node);

		lock.unset();
		return cacheData + it->second->element*offsetCube;
			
	}
	else // If not exists
	{
		index_node_t 	 removedCube = (index_node_t)0;
		NodeLinkedList * node = queuePositions->getFirstFreePosition(idCube, &removedCube);

		if (node != NULL)
		{
			indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
			if (removedCube!= (index_node_t)0)
				indexStored.erase(indexStored.find(removedCube));

			unsigned pos   = node->element;

			queuePositions->moveToLastPosition(node);
			queuePositions->addReference(node);

			fileManager->readCube(idCube, cacheData+ pos*offsetCube);

			lock.unset();

			return cacheData+ pos*offsetCube;
		}
		else // there is no free slot
		{
			lock.unset();
			return NULL; 
		}
	}
}

void cubeCacheCPU::pop_cube(index_node_t idCube)
{
#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif
	lock.set();

	// Find the cube in the CPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist remove reference
	{
		NodeLinkedList * node = it->second;
		queuePositions->removeReference(node);
	}
	else
	{
		lock.unset();
		std::cerr<<"Cache is unistable"<<std::endl;
		throw;
	}
	lock.unset();
}
}
