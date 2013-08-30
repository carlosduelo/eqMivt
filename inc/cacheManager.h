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
		std::map<uint32_t , eqMivt::cubeCache *>	_caches;
		std::map<uint32_t , int>					_ids;
		lunchbox::Lock								_lock;

		vmml::vector<3, int>	_cubeDim;
		int	_cubeInc;
		int	_levelCube;
		int	_nLevels;
		int _levelCubeCPU;
		int _numElements;

	public:
		CacheManager();

		~CacheManager();
		
		bool init(std::string type_file, std::vector<std::string> file_params, std::string octree_file_name, int cubeInc);
		bool reSize(int levelCube, int nLevels, int numElements, int numElementsCPU, int levelCubeCPU);	
		bool setOffset(vmml::vector<3, int> offset);
		bool checkStatus(CacheHandler * cacheHandler);
		bool forceResize(CacheHandler * cacheHandler);
		bool getCache(uint32_t device, CacheHandler * cacheHandler );
};
}

#endif /*EQ_MIVT_CUBE_MANAGER_H*/
