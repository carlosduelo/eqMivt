/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "cacheManager.h"

namespace eqMivt
{
CacheManager::CacheManager()
{
	_cubeCacheCPU = 0;
	_cubeDim.set(0,0,0);
	_cubeInc = 0;
	_levelCube = -1;
	_nLevels = 0;
	_numElements = 0;
	_levelCubeCPU = -1;
}

CacheManager::~CacheManager()
{
	if (_cubeCacheCPU != 0)
		delete _cubeCacheCPU;
	for (std::map<uint32_t, eqMivt::cubeCache *>::iterator it = _caches.begin(); it!=_caches.end(); it++)
		delete it->second;
}

bool CacheManager::init(std::string type_file, std::vector<std::string> file_params, std::string octree_file_name, int cubeInc)
{
	_lock.set();
	bool result = true;

	if (_cubeCacheCPU != 0)
	{
		result = false;
	}
	else
	{
		_cubeCacheCPU = new cubeCacheCPU();
		result = _cubeCacheCPU->init(type_file, file_params, octree_file_name);
		_cubeInc = cubeInc;
	}
	_lock.unset();

	return result;
}


bool CacheManager::reSize(int levelCube, int nLevels, int numElements, int numElementsCPU, int levelCubeCPU)
{
	bool result = true;
	_lock.set();
	if (_levelCube != levelCube || levelCubeCPU != _levelCubeCPU || nLevels != _nLevels)
	{
		int dimC = exp2(nLevels - levelCubeCPU);
		vmml::vector<3, int> cubeDimCPU(dimC, dimC, dimC);
		result = _cubeCacheCPU->reSize(cubeDimCPU, _cubeInc, levelCubeCPU, nLevels, numElementsCPU);

		_nLevels = nLevels;
		_levelCube = levelCube;
		_levelCubeCPU = levelCubeCPU;
		int d = exp2(_nLevels - levelCube);
		_cubeDim.set(d,d,d);
		_numElements = numElements;
	}
	_lock.unset();

	return result;
}

bool CacheManager::setOffset(vmml::vector<3, int> offset)
{
	return _cubeCacheCPU->setOffset(offset);
}

bool CacheManager::forceResize(CacheHandler * cacheHandler)
{
	_lock.set();
	bool result = cacheHandler->forceResize();	
	_lock.unset();
	return result; 
}

bool CacheManager::checkStatus(CacheHandler * cacheHandler)
{
	_lock.set();
	bool result = cacheHandler->isValid() && cacheHandler->reSize(_cubeDim, _cubeInc, _levelCube, _numElements);	
	_lock.unset();
	return result; 
}

bool CacheManager::getCache(uint32_t device, CacheHandler * cacheHandler )
{
	_lock.set();

	std::map<uint32_t, eqMivt::cubeCache *>::iterator itC;
	itC = _caches.find(device);
	int result = true;

	// Create the cache
	if (itC == _caches.end())
	{
		eqMivt::cubeCache * c = new cubeCache();
		if (c->init(_cubeCacheCPU, MAX_WORKERS, device)) //&& c->reSize(_cubeDim, _cubeInc, _levelCube, _numElements))
		{
			_caches[device]  = c;
			_ids[device] = 0;
			cacheHandler->set(c, 0);
			result = true;
		}
		else
		{
			delete c;
			result = false;
		}
	}
	else
	{
		_ids[device] += 1;
		cacheHandler->set(itC->second, _ids[device]); 
		result = true;
	}

	_lock.unset();

	return result;
}

}
