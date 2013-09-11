/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_CUBE_CAHCE_CPU_H
#define EQ_MIVT_CUBE_CAHCE_CPU_H

#include <typedef.h>
#include <fileFactory.h>
#include <linkedList.h>

#include <lunchbox/lock.h>

#include <boost/unordered_map.hpp>

namespace eqMivt
{
class cubeCacheCPU 
{
	private:
		lunchbox::Lock	_lock;

		index_node_t			_minIndex;
		index_node_t			_maxIndex;
		vmml::vector<3, int>	_cubeDim;
		vmml::vector<3, int>	_cubeInc;
		vmml::vector<3, int>	_realcubeDim;
		int	_offsetCube;
		int	_levelCube;
		int	_nLevels;

		boost::unordered_map<index_node_t, NodeLinkedList *>	_indexStored;
		LinkedList	*											_queuePositions;

		double			_memoryCPU;
		int				_maxElements;
		float		*	_cacheData;

		std::vector<index_node_t> _pendingCubes;

		// Acces to file
		FileManager	*	_fileManager;

	public:

		cubeCacheCPU();
		~cubeCacheCPU();

		bool init(std::string type_file, std::vector<std::string> file_params, std::string octree_file_name);
		
		bool reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int nLevels, int numElements );

		bool setOffset(vmml::vector<3, int> offset);

		vmml::vector<3, int>    getCubeDim(){ return _cubeDim; }
		vmml::vector<3, int>    getCubeInc(){ return _cubeInc; }
		vmml::vector<3, int>    getRealCubeDim(){ return _realcubeDim; }
		int						getLevelCube(){ return _levelCube;}
		int						getnLevels(){ return _nLevels; }

		float *  push_cube(index_node_t  idCube);

		float *  push_cubeBuffered(index_node_t  idCube, bool * pending);

		float * getPointerCube(index_node_t  idCube);

		void	readBufferCubes();

		void pop_cube(index_node_t idCube);
};

}

#endif /*EQ_MIVT_CUBE_CAHCE_CPU_H*/
