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

		vmml::vector<3, int>	_cubeDim;
		vmml::vector<3, int>	_cubeInc;
		vmml::vector<3, int>	_realcubeDim;
		int	_offsetCube;
		int	_levelCube;
		int	_nLevels;

		boost::unordered_map<index_node_t, NodeLinkedList *>	_indexStored;
		LinkedList	*											_queuePositions;

		int				_maxElements;
		float		*	_cacheData;

		// Acces to file
		FileManager	*	_fileManager;

	public:

		cubeCacheCPU();
		~cubeCacheCPU();

		bool init(std::string type_file, std::vector<std::string> file_params, int nLevels);
		
		bool reSize(vmml::vector<3, int> cubeDim, int cubeInc, int levelCube, int numElements );

		vmml::vector<3, int>    getCubeDim(){ return _cubeDim; }
		vmml::vector<3, int>    getCubeInc(){ return _cubeInc; }
		vmml::vector<3, int>    getRealCubeDim(){ return _realcubeDim; }
		int						getLevelCube(){ return _levelCube;}
		int						getnLevels(){ return _nLevels; }

		float *  push_cube(index_node_t  idCube);

		void pop_cube(index_node_t idCube);
};

}

#endif /*EQ_MIVT_CUBE_CAHCE_CPU_H*/
