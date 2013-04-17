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

#ifdef _BUNORDER_MAP_
	#include <boost/unordered_map.hpp>
#else
	#include <map>
#endif

namespace eqMivt
{
class cubeCacheCPU 
{
	private:
		lunchbox::Lock	lock;

		vmml::vector<3, int>	cubeDim;
		vmml::vector<3, int>	cubeInc;
		vmml::vector<3, int>	realcubeDim;
		int	offsetCube;
		int	levelCube;
		int	nLevels;

		#ifdef _BUNORDER_MAP_
			boost::unordered_map<index_node_t, NodeLinkedList *> indexStored;
		#else
			std::map<index_node_t, NodeLinkedList *> indexStored;
		#endif

		LinkedList	*	queuePositions;

		int			maxElements;
		float		*	cacheData;

		// Acces to file
		FileManager	*	fileManager;

	public:

		bool init(std::string type_file, std::vector<std::string> file_params, int p_maxElements, vmml::vector<3, int> p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels);
		~cubeCacheCPU();

		vmml::vector<3, int>    getCubeDim(){ return cubeDim; }
		vmml::vector<3, int>    getCubeInc(){ return cubeInc; }
		int     		getLevelCube(){ return levelCube;}
		int			getnLevels(){ return nLevels; }

		float *  push_cube(index_node_t  idCube);

		void pop_cube(index_node_t idCube);
};

}

#endif /*EQ_MIVT_CUBE_CAHCE_CPU_H*/
