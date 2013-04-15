/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_OCTREE_CONTAINER_H
#define EQ_MIVT_OCTREE_CONTAINER_H

#include <eq/eq.h>

#include <typedef.h>

namespace eqMivt
{


class OctreeContainer
{
	private:
		float 			_isosurface;
		vmml::vector<3, int>	_realDim;
		int 			_dimension;
		int 			_nLevels;
		int 			_maxLevel;

		index_node_t ** 	_octree;
		index_node_t * 		_memoryGPU;
		int	*		_sizes;
		int			_device;
	public:
		/* Lee el Octree de un fichero */
		OctreeContainer(int device);

		~OctreeContainer();

		static int getnLevelsFromOctreeFile(std::string file_name);

		bool readOctreeFile(std::string file_name, int p_maxLevel);

		int getDevice(){ return _device; }

		int getnLevels(){ return _nLevels; }

		int getMaxLevel(){ return _maxLevel; }

		float getIsosurface(){ return _isosurface; }

		index_node_t ** getOctree(){ return _octree; }

		int *		getSizes(){ return _sizes;}
};
}
#endif /*EQ_MIVT_OCTREE_CONTAINER_H*/
