/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_OCTREE_MANAGER_H
#define EQ_MIVT_OCTREE_MANAGER_H

#include <typedef.h>
#include <octree.h>

#include <eq/eq.h>
#include <lunchbox/lock.h>

#include <map>
#include <string>
#include <iostream>
#include <fstream>

namespace eqMivt
{
class OctreeManager
{
	private:
		std::ifstream _file;

		lunchbox::Lock								_lock;
		/* General parameters */
		int _nLevels;
		int	_maxLevel;
		int _dimension;
		vmml::vector<3, int> _realDim;
		int _numOctrees;

		float * _isosurfaces;
		int ** _sizes;
		int * _desp;
		int ** _numCubes;
		int * _maxHeight;
		int * _cubeCacheLevel;
		index_node_t ** _octreeData;
		int _currentOctree;

		std::map<uint32_t , Octree *>   _octrees;

		/* private methods */
		void _readCurrentOctree();
		void _setBestCubeLevel();
	public:
		OctreeManager();

		~OctreeManager();

		static int readNLevelsFromFile(std::string file_name);
		static int readMaxLevelsFromFile(std::string file_name);
		static int readDimensionFromFile(std::string file_name);
		static vmml::vector<3, int> readRealDimFromFile(std::string file_name);
		static int readNumOctreesFromFile(std::string file_name);

		bool init(std::string file_name);

		int getNLevels() { return _nLevels; }
		int getMaxLevel() { return _maxLevel; }
		int getDimension() { return _dimension; }
		vmml::vector<3, int> getRealDim() { return _realDim; }
		vmml::vector<3, int> getCurrentRealDim() { return vmml::vector<3, int>(_realDim.x(), _maxHeight[_currentOctree], _realDim.z()); }
		int getNumOctrees() { return _numOctrees; }
		float getCurretIsosurface() { return _isosurfaces[_currentOctree]; }
		int	getMaxHeight() { return _maxHeight[_currentOctree]; }
		int getBestCubeLevel(){ return _cubeCacheLevel[_currentOctree]; }

		bool setCurrentOctree(int currentOctree);
		bool checkStatus(uint32_t device);

		Octree * getOctree(uint32_t device);
};
}
#endif /*EQ_MIVT_OCTREE_MANAGER_H*/
