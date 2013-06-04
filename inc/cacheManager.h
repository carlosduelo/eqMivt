/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_CUBE_MANAGER_H
#define EQ_MIVT_CUBE_MANAGER_H

#include "cubeCache.h"
#include "cubeCacheCPU.h"

#include <map>
#include <lunchbox/lock.h>

#define MAX_WORKERS 32

namespace eqMivt
{

class CacheManager
{
	private:
		cubeCacheCPU		*						_cubeCacheCPU;
		std::map<int , eqMivt::cubeCache *>			_caches;
		std::map<int , int>							_ids;
		lunchbox::Lock								_lock;

		vmml::vector<3, int>	_cubeDim;
		int	_cubeInc;
		int	_levelCube;
		int	_nLevels;
		int _levelDif;
		int _numElements;

	public:
		CacheManager();

		~CacheManager();
		
		bool init(std::string type_file, std::vector<std::string> file_params, int nLevels);
		bool reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int numElements, int numElementsCPU, int levelDif);	

		CacheHandler getCache(int device);
};
}

#endif /*EQ_MIVT_CUBE_MANAGER_H*/
