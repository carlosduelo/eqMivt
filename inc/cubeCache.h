/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */


#ifndef EQ_MIVT_CUBE_CACHE_H
#define EQ_MIVT_CUBE_CACHE_H

#include "cubeCacheGPU.h"

namespace eqMivt
{

	typedef struct 
	{ 
		index_node_t    cubeID; 
		float   *       data; 
		int             state; 
	} cacheElement_t; 


class cubeCache
{
	private:
		cubeCacheGPU _cache;

		boost::unordered_map<index_node_t, cacheElement_t > * _insertedCubes;

	public:
		~cubeCache();

		bool init(cubeCacheCPU * cpuCache, int numWorkers, uint32_t device);

		bool reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int numElements);

		int getCacheLevel() {return _cache.getLevelCube(); }

		vmml::vector<3, int> getCubeDim(){ return _cache.getCubeDim(); }

		vmml::vector<3, int> getCubeInc(){ return _cache.getCubeInc(); }

		// Return true if exist some petition and false otherwise
		bool push(visibleCube_t * visibleCubes, int * indexCube, int * num, int octreeLevel, int threadID, cudaStream_t stream);

		void pop(visibleCube_t * visibleCubes, int * indexCube, int num, int octreeLevel, int threadID, cudaStream_t stream);
};

class CacheHandler
{
	private:
		cubeCache * _cache;
		int			_id;
	public:
		CacheHandler()
		{
			_cache = 0;
			_id = -1;
		}
		void set(cubeCache * cache, int id)
		{
			_cache = cache;
			_id = id;
		}
		bool isValid() { return _cache != 0; }
		int getCacheLevel() {return _cache->getCacheLevel(); }
		vmml::vector<3, int> getCubeDim(){ return _cache->getCubeDim(); }
		vmml::vector<3, int> getCubeInc(){ return _cache->getCubeInc(); } 
		bool reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int numElements)
		{
			return _cache->reSize(cubeDim, cubeInc, levelCube, numElements);
		}
		bool push(visibleCube_t * visibleCubes, int * indexCube, int * num, int octreeLevel, cudaStream_t stream)
		{
			_cache->push(visibleCubes, indexCube, num, octreeLevel, _id, stream);
		}

		void pop(visibleCube_t * visibleCubes, int * indexCube, int num, int octreeLevel, cudaStream_t stream)
		{
			_cache->pop(visibleCubes, indexCube, num, octreeLevel, _id, stream);
		}
};

}

#endif /*EQ_MIVT_CUBE_CACHE_H*/
