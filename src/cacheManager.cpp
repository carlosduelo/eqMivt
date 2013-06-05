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
	_levelCube = 0;
	_nLevels = 0;
	_numElements = 0;
	_levelDif = 0;
}

CacheManager::~CacheManager()
{
	if (_cubeCacheCPU != 0)
		delete _cubeCacheCPU;
	for (std::map<int , eqMivt::cubeCache *>::iterator it = _caches.begin(); it!=_caches.end(); it++)
		delete it->second;
}

bool CacheManager::init(std::string type_file, std::vector<std::string> file_params, int nLevels)
{
	_lock.set();
	bool result = true;
	if (_cubeCacheCPU != 0)
	{
		result = _cubeCacheCPU->init(type_file, file_params, nLevels);
		_nLevels = nLevels;
	}
	_lock.unset();

	return result;
}


bool CacheManager::reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int numElements, int numElementsCPU, int levelDif)
{
	bool result = true;
	_lock.set();
	if (_levelCube != levelCube || levelDif != _levelDif)
	{
		int levelCubeCPU = levelCube - levelDif; 
		int dimC = exp2(_cubeCacheCPU->getnLevels() - levelCubeCPU);
		vmml::vector<3, int> cubeDimCPU(dimC, dimC, dimC);
		result = result && _cubeCacheCPU->reSize(cubeDimCPU, cubeInc, levelCubeCPU, numElementsCPU);

		for (std::map<int , eqMivt::cubeCache *>::iterator it = _caches.begin(); it!=_caches.end(); it++)
			result = result && it->second->reSize(cubeDim, cubeInc,levelCube,numElements);	

			_cubeDim = cubeDim;
			_cubeInc = cubeInc;
			_levelCube = levelCube;
			_numElements = numElements;
	}
	_lock.unset();

	return result;
}

CacheHandler CacheManager::getCache(int device)
{
	CacheHandler result; 
	_lock.set();

	std::map<int , eqMivt::cubeCache *>::iterator itC;
	itC = _caches.find(device);

	// Create the cache
	if (itC == _caches.end())
	{
		eqMivt::cubeCache * c = new cubeCache();
		if (c->init(_cubeCacheCPU, MAX_WORKERS) && c->reSize(_cubeDim, _cubeInc, _levelCube, _numElements))
		{
			_caches[device]  = c;
			_ids[device] = 0;
			result.set(c, 0);
		}
		else
			result.set(0,0);
	}
	else
	{
		_ids[device] += 1;
		result.set(itC->second, _ids[device]); 
	}

	_lock.unset();

	return result;
}

}
