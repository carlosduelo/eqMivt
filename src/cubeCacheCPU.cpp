/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <cubeCacheCPU.h>

#include <cuda_runtime.h>
#include "memoryCheck.h"
#include "mortonCodeUtil_CPU.h"

#define MAX_SIZE 512*1024*1024

namespace eqMivt
{

cubeCacheCPU::cubeCacheCPU()
{
		_cubeDim.set(0,0,0);
		_cubeInc.set(0,0,0);
		_realcubeDim.set(0,0,0);
		_offsetCube = 0;
		_levelCube = 0;
		_nLevels = 0;

		_indexStored;
		_queuePositions = 0;

		_minIndex = 0;
		_maxIndex = 0;

		_memoryCPU = -10.0;
		_maxElements = 0;
		_cacheData = 0;
		
		_fileManager = 0;
}

cubeCacheCPU::~cubeCacheCPU()
{
	if (_fileManager != 0)
		delete _fileManager;
	if (_queuePositions != 0)
		delete _queuePositions;
	if (_cacheData != 0)
		cudaFreeHost(_cacheData);
}

bool cubeCacheCPU::init(std::string type_file, std::vector<std::string> file_params, std::string octree_file_name)
{
	if (_fileManager != 0)
		return false;

	// OpenFile
	_fileManager = eqMivt::CreateFileManage(type_file, file_params);
	if (_fileManager == 0)
	{
		LBERROR<<"Cache CPU: error initialization file"<<std::endl;
		return false;
	}

	return _fileManager->checkInit(octree_file_name);
}

bool cubeCacheCPU::reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int nLevels, int numElements)
{
	if (levelCube == _levelCube && _nLevels == nLevels)
		return true;

	// cube size
	_cubeDim 	= cubeDim;
	_cubeInc.set(cubeInc,cubeInc,cubeInc);
	_realcubeDim	= cubeDim + 2 * cubeInc;
	_levelCube	= levelCube;
	_nLevels = nLevels;
	_offsetCube	= (_cubeDim.x()+2*_cubeInc.x())*(_cubeDim.y()+2*_cubeInc.y())*(_cubeDim.z()+2*_cubeInc.z());
	_minIndex = coordinateToIndex(vmml::vector<3, int>(0,0,0), _levelCube, _nLevels); 
	int d = exp2(_nLevels);
	_maxIndex = coordinateToIndex(vmml::vector<3, int>(d-1,d-1,d-1), _levelCube, _nLevels);

	if (_cacheData != 0)
	{
		if (cudaSuccess != cudaFreeHost((void*)_cacheData))
		{
			LBERROR<<"Cache CPU: Error creating cpu cache: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			return false;
		
		}
		_cacheData = 0;
	}

	_memoryCPU = getMemorySize();
	if (_memoryCPU == 0)
	{
		LBERROR<<"Not possible, check memory aviable (the call failed due to OS limitations)"<<std::endl;
		_memoryCPU = MAX_SIZE; 
	}
	else
	{
		_memoryCPU *= 0.8;
	}

	double cd = _offsetCube;
	cd *= sizeof(float);
	_maxElements = _memoryCPU/cd;
	if (_maxElements == 0)
	{
		LBERROR<<"Cache CPU: Memory aviable is not enough "<<_memoryCPU/1024/1024<<" MB"<<std::endl;
		return false;
	}

	if (numElements != 0)
	{
		if (numElements > _maxElements)
		{
			LBERROR<<"Cache CPU: max elements in cache cpu are to big"<<std::endl;
			return false;
		}
		_maxElements = numElements;
	}

	if (_cacheData == 0)
	{
		LBINFO<<"Creating cache in CPU: "<< _memoryCPU/1024.0f/1024.0f<<" MB: "<<std::endl;
		if (cudaSuccess != cudaHostAlloc((void**)&_cacheData, _memoryCPU, cudaHostAllocDefault))
		{
			LBERROR<<"Cache CPU: Error creating cpu cache: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			return false;
		
		}
	}

	_indexStored.clear();

	if (_queuePositions != 0)
		delete _queuePositions;
	_queuePositions	= new LinkedList(_maxElements);


	return true;
}

bool cubeCacheCPU::setOffset(vmml::vector<3, int> offset)
{
	_lock.set();
	_fileManager->setOffset(offset);
	_lock.unset();

	return true;
}


float * cubeCacheCPU::push_cube(index_node_t idCube)
{
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;

	if (idCube < _minIndex || idCube > _maxIndex)
	{
		LBERROR<<"Cache CPU: Trying to push a worng index cube "<<idCube<<std::endl;
		return 0;
	}

	_lock.set();
	
	// Find the cube in the CPU cache
	it = _indexStored.find(idCube);
	if ( it != _indexStored.end() ) // If exist
	{
		NodeLinkedList * node = it->second;

		_queuePositions->addReference(node, idCube);

		_lock.unset();
		return _cacheData + it->second->element*_offsetCube;
			
	}
	else // If not exists
	{
		index_node_t 	 removedCube = (index_node_t)0;
		NodeLinkedList * node = _queuePositions->getFirstFreePosition(idCube, &removedCube);

		if (node != 0)
		{
			_indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
			if (removedCube!= (index_node_t)0)
				_indexStored.erase(_indexStored.find(removedCube));

			unsigned pos   = node->element;

			_queuePositions->addReference(node, idCube);

			_fileManager->readCube(idCube, _cacheData+ pos*_offsetCube, _levelCube, _nLevels, _cubeDim, _cubeInc, _realcubeDim);

			_lock.unset();

			return _cacheData+ pos*_offsetCube;
		}
		else // there is no free slot
		{
			_lock.unset();
			return 0; 
		}
	}
}

void cubeCacheCPU::pop_cube(index_node_t idCube)
{
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;

	_lock.set();

	// Find the cube in the CPU cache
	it = _indexStored.find(idCube);
	if ( it != _indexStored.end() ) // If exist remove reference
	{
		NodeLinkedList * node = it->second;
		_queuePositions->removeReference(node);
	}
	else
	{
		_lock.unset();
		LBERROR<<"Cache is unistable"<<std::endl;
		throw;
	}
	_lock.unset();
}

float * cubeCacheCPU::getPointerCube(index_node_t  idCube)
{
	_lock.set();

	float * d = 0;
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
	// Find the cube in the CPU cache
	it = _indexStored.find(idCube);
	if ( it != _indexStored.end() ) // If exist
		d =  _cacheData + it->second->element*_offsetCube;

	_lock.unset();
	return d;
}


float *  cubeCacheCPU::push_cubeBuffered(index_node_t  idCube, bool * pending)
{
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;

	if (idCube < _minIndex || idCube > _maxIndex)
	{
		LBERROR<<"Cache CPU: Trying to push a worng index cube "<<idCube<<std::endl;
		return 0;
	}

	_lock.set();

	// Find the cube in the CPU cache
	it = _indexStored.find(idCube);
	if ( it != _indexStored.end() ) // If exist
	{
		NodeLinkedList * node = it->second;

		_queuePositions->addReference(node, idCube);

		if (std::find(_pendingCubes.begin(), _pendingCubes.end(), idCube) != _pendingCubes.end())
			*pending = true;
		else
			*pending = false;

		_lock.unset();
		return _cacheData + it->second->element*_offsetCube;
			
	}
	else // If not exists
	{
		index_node_t 	 removedCube = (index_node_t)0;
		NodeLinkedList * node = _queuePositions->getFirstFreePosition(idCube, &removedCube);

		if (node != 0)
		{
			_indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
			if (removedCube!= (index_node_t)0)
				_indexStored.erase(_indexStored.find(removedCube));

			unsigned pos   = node->element;

			_queuePositions->addReference(node, idCube);

			_fileManager->addCubeToBuffer(idCube, _cacheData+ pos*_offsetCube, _levelCube, _nLevels, _cubeDim, _cubeInc, _realcubeDim);

			_pendingCubes.push_back(idCube);
			*pending = true;
			_lock.unset();
			return _cacheData+ pos*_offsetCube;
		}
		else // there is no free slot
		{
			_lock.unset();
			return 0; 
		}
	}
}

void	cubeCacheCPU::readBufferCubes()
{
	_lock.set();
	_fileManager->readBufferedCubes();
	_pendingCubes.clear();
	_lock.unset();
}

}
