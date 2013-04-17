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
		cubeCacheGPU cache;

#ifdef _BUNORDER_MAP_
		boost::unordered_map<index_node_t, cacheElement_t > * insertedCubes;
#else
		std::map<index_node_t, cacheElement_t > * insertedCubes;
#endif

	public:
		bool init(cubeCacheCPU * p_cpuCache, int p_numWorkers, int p_maxElements);

		~cubeCache();

		int getCacheLevel() {return cache.getLevelCube(); }

		vmml::vector<3, int> getCubeDim(){ return cache.getCubeDim(); }

		vmml::vector<3, int> getCubeInc(){ return cache.getCubeInc(); }

		void push(visibleCube_t * visibleCubes, int num, int octreeLevel, int threadID, cudaStream_t stream);

		void pop(visibleCube_t * visibleCubes, int num, int octreeLevel, int threadID, cudaStream_t stream);
};

}

#endif /*EQ_MIVT_CUBE_CACHE_H*/
