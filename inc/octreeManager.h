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

		lunchbox::Lock		_lock;
		/* General parameters */
		int _lastLevel; 
		int * _nLevels;
		int	* _maxLevel;
		int * _dimension;
		vmml::vector<3, int> * _realDim;
		vmml::vector<3, int> * _startC;
		vmml::vector<3, int> * _finishC;
		int _numOctrees;
		vmml::vector<3, int> _realDimensionVolume;
		double * _xGrid;
		double * _yGrid;
		double * _zGrid;

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

		static int readNumOctreesFromFile(std::string file_name);

		bool init(std::string file_name);

		int getNLevels() { return _nLevels[_currentOctree]; }
		int getMaxLevel() { return _maxLevel[_currentOctree]; }
		int getDimension() { return _dimension[_currentOctree]; }
		vmml::vector<3, int> getRealDimVolumeData() { return _realDimensionVolume; }
		vmml::vector<3, float> getRealDimVolume() ;
		vmml::vector<3, float> getCurrentStartCoord();
		vmml::vector<3, float> getCurrentFinishCoord();
		int getNumOctrees() { return _numOctrees; }
		float getCurretIsosurface() { return _isosurfaces[_currentOctree]; }
		int	getMaxHeight() { return _maxHeight[_currentOctree]; }
		int getBestCubeLevel(){ return _cubeCacheLevel[_currentOctree]; }
		double * getxGrid();
		double * getyGrid();
		double * getzGrid();
		vmml::vector<3, int> getCurrentOffset(){ return _startC[_currentOctree]; }

		bool setCurrentOctree(int currentOctree);
		bool checkStatus(uint32_t device);

		Octree * getOctree(uint32_t device);
};
}
#endif /*EQ_MIVT_OCTREE_MANAGER_H*/
