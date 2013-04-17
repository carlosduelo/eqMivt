/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_CUBE_CACHE_GPU_H
#define EQ_MIVT_CUBE_CACHE_GPU_H

#include "cubeCacheCPU.h"


#include "cuda_runtime.h"

namespace eqMivt
{

class cubeCacheGPU
{
	private:
		// Acces to file
		cubeCacheCPU *        	cpuCache;

		lunchbox::Lock			lock;

		vmml::vector<3, int>    cubeDim;
		vmml::vector<3, int>    cubeInc;
		vmml::vector<3, int>    realcubeDim;
		int						offsetCube;
		int						levelCube;
		int						nLevels;

#ifdef _BUNORDER_MAP_
		boost::unordered_map<index_node_t, NodeLinkedList *> indexStored;
#else
		std::map<index_node_t, NodeLinkedList *> indexStored;
#endif

		LinkedList      *       queuePositions;

		int                     maxElements;
		float           *       cacheData;

	public:
		cubeCacheGPU();

		~cubeCacheGPU();

		vmml::vector<3, int>    getCubeDim(){ return cubeDim; }
		vmml::vector<3, int>    getCubeInc(){ return cubeInc; }
		int						getLevelCube(){ return levelCube;}
		int						getnLevels(){ return nLevels; }

		bool init(cubeCacheCPU * p_cpuCache, int p_maxElements);

		float * push_cube(index_node_t idCube, cudaStream_t stream);

		void  pop_cube(index_node_t idCube);
};

}

#endif /*EQ_MIVT_CUBE_CACHE_GPU_H*/
