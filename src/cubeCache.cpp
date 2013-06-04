/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "cubeCache.h"

namespace eqMivt
{

cubeCache::~cubeCache()
{
	delete[] _insertedCubes;
}

bool cubeCache::init(cubeCacheCPU * cpuCache, int numWorkers)
{
	_insertedCubes = new boost::unordered_map<index_node_t, cacheElement_t >[numWorkers];

	return _cache.init(cpuCache);

}

bool cubeCache::reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int numElements)
{
	return _cache.reSize(cubeDim, cubeInc, levelCube, numElements);
}


bool cubeCache::push(visibleCube_t * visibleCubes, int *indexCube, int * num, int octreeLevel, int threadID, cudaStream_t stream)
{
	boost::unordered_map<index_node_t, cacheElement_t>::iterator it;

	int nextIndex	= 0;

	// For each visible cube push into the cache
	for(int i=0; i<(*num); i++)
	{
		int index = indexCube[i];
		if (visibleCubes[index].state != PAINTED)
		{
			if (visibleCubes[index].state == NOCACHED || visibleCubes[index].state == CUBE)
			{
				index_node_t idCube = visibleCubes[index].id >> (3*(octreeLevel - _cache.getLevelCube()));

				it = _insertedCubes[threadID].find(idCube);
				if (it == _insertedCubes[threadID].end()) // If does not exist, do not push again
				{
					float * cubeData = _cache.push_cube(idCube, stream);

					visibleCubes[index].cubeID  = idCube;
					visibleCubes[index].state   = cubeData == 0 ? NOCACHED : CACHED;
					visibleCubes[index].data    = cubeData;

					cacheElement_t newCube;
					newCube.cubeID = idCube;
					newCube.state = cubeData == 0 ? NOCACHED : CACHED;
					newCube.data = cubeData;

					_insertedCubes[threadID].insert(std::pair<index_node_t, cacheElement_t>(idCube, newCube));
				}
				else
				{
					visibleCubes[index].cubeID  = it->second.cubeID;
					visibleCubes[index].state   = it->second.state;
					visibleCubes[index].data    = it->second.data;

				}

			}
			indexCube[nextIndex] = index;
			nextIndex++;
		}
	}

	*num = nextIndex;
	return nextIndex != 0;

}

void cubeCache::pop(visibleCube_t * visibleCubes, int * indexCube, int num, int octreeLevel, int threadID, cudaStream_t stream)
{
	boost::unordered_map<index_node_t, cacheElement_t>::iterator it;

	it = _insertedCubes[threadID].begin();

	while(it != _insertedCubes[threadID].end())
	{
		if (it->second.state == CACHED)
		{
			_cache.pop_cube(it->second.cubeID);
		}
		it++;
	}

	_insertedCubes[threadID].clear();

}

}
