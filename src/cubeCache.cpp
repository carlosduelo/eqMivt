/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "cubeCache.h"

namespace eqMivt
{


bool cubeCache::init(cubeCacheCPU * p_cpuCache, int p_numWorkers, int p_maxElements)
{
#ifdef _BUNORDER_MAP_
	insertedCubes = new boost::unordered_map<index_node_t, cacheElement_t >[p_numWorkers];
#else
	insertedCubes = new std::map<index_node_t, cacheElement_t >[p_numWorkers];
#endif

	return cache.init(p_cpuCache, p_maxElements);	

}

cubeCache::~cubeCache()
{
	delete[] insertedCubes;
}

bool cubeCache::push(visibleCube_t * visibleCubes, int num, int octreeLevel, int threadID, cudaStream_t stream)
{
#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, cacheElement_t>::iterator it;
#else
	std::map<index_node_t, cacheElement_t>::iterator it;
#endif

	bool notEnd = false;

	// For each visible cube push into the cache
	for(int i=0; i<num; i++)
	{
		if (visibleCubes[i].state == NOCACHED || visibleCubes[i].state == CUBE)
		{
			notEnd = true;
			index_node_t idCube = visibleCubes[i].id >> (3*(octreeLevel - cache.getLevelCube()));

			it = insertedCubes[threadID].find(idCube);
			if (it == insertedCubes[threadID].end()) // If does not exist, do not push again
			{
				float * cubeData = cache.push_cube(idCube, stream);

				visibleCubes[i].cubeID  = idCube;
				visibleCubes[i].state   = cubeData == 0 ? NOCACHED : CACHED;
				visibleCubes[i].data    = cubeData;

				cacheElement_t newCube;
				newCube.cubeID = idCube;
				newCube.state = cubeData == 0 ? NOCACHED : CACHED;
				newCube.data = cubeData;

				insertedCubes[threadID].insert(std::pair<index_node_t, cacheElement_t>(idCube, newCube));
			}
			else
			{
				visibleCubes[i].cubeID  = it->second.cubeID;
				visibleCubes[i].state   = it->second.state;
				visibleCubes[i].data    = it->second.data;

			}
		}
		else if  (visibleCubes[i].state != PAINTED)
			notEnd = true;
	}

	return notEnd;

}

void cubeCache::pop(visibleCube_t * visibleCubes, int num, int octreeLevel, int threadID, cudaStream_t stream)
{
#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, cacheElement_t>::iterator it;
#else
	std::map<index_node_t, cacheElement_t>::iterator it;
#endif

	it = insertedCubes[threadID].begin();

	while(it != insertedCubes[threadID].end())
	{
		if (it->second.state == CACHED)
		{
			cache.pop_cube(it->second.cubeID);
		}
		it++;
	}

	insertedCubes[threadID].clear();

}

}
