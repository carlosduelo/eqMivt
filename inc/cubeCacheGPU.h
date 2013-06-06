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
		cubeCacheCPU *        	_cpuCache;

		lunchbox::Lock			_lock;

		vmml::vector<3, int>    _cubeDim;
		vmml::vector<3, int>    _cubeInc;
		vmml::vector<3, int>    _realcubeDim;
		int						_offsetCube;
		int						_levelCube;
		int						_nLevels;

		boost::unordered_map<index_node_t, NodeLinkedList *> _indexStored;

		LinkedList      *       _queuePositions;

		uint32_t				_device;
		int                     _maxElements;
		float           *       _cacheData;

	public:
		cubeCacheGPU();

		~cubeCacheGPU();

		vmml::vector<3, int>    getCubeDim(){ return _cubeDim; }
		vmml::vector<3, int>    getCubeInc(){ return _cubeInc; }
		int						getLevelCube(){ return _levelCube;}
		int						getnLevels(){ return _nLevels; }

		bool init(cubeCacheCPU * cpuCache, uint32_t device);

		bool reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int numElements);

		float * push_cube(index_node_t idCube, cudaStream_t stream);

		void  pop_cube(index_node_t idCube);
};

}

#endif /*EQ_MIVT_CUBE_CACHE_GPU_H*/
