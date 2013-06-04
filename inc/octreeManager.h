/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_OCTREE_MANAGER_H
#define EQ_MIVT_OCTREE_MANAGER_H

#include <typedef.h>

#include <string>
#include <iostream>
#include <fstream>

namespace eqMivt
{
class OctreeManager
{
	private:
		std::ifstream _file;

		/* General parameters */
		int _nLevels;
		int	_maxLevel;
		int _dimension;
		vmml::vector<3, int> _realDim;
		int _numOctrees;

		float * _isosurfaces;
		int ** _sizes;
		int * _desp;
		int * _numCubes;
		int * _maxHeight;
		int * _cubeCacheLevel;
		float ** _octreeData;
		int _currentOctree;

		/* private methods */
		bool _checkOctreeFileValid(std::ifstream * file);
		void _readCurrentOctree();
	public:
		OctreeManager();

		~OctreeManager();

		static int readNLevelsFromFile(std::string file_name);
		static int readMaxLevelsFromFile(std::string file_name);
		static vmml::vector<3, int> readDimensionFromFile(std::string file_name);
		static int readRealDimFromFile(std::string file_name);
		static int readNumOctreesFromFile(std::string file_name);

		bool init(std::string file_name);

		int getNLevels() { return _nLevels; }
		int getMaxLevel() { return _maxLevel; }
		int getDimension() { return _dimension; }
		vmml::vector<3, int> getRealDim() { return _realDim; }
		int getNumOctrees() { return _numOctrees; }
		float getCurretIsosurface() { return _isosurfaces[_currentOctree]; }
		int	getMaxHeight() { return _maxHeight[_currentOctree]; }
		int getBestCubeLevel(){ return _cubeCacheLevel[_currentOctree]; }

		#if 0
		void setCurrentOctree(int currentOctree);
		void cpyOctreeToDevice(int device);
		#endif
};
}
#endif /*EQ_MIVT_OCTREE_MANAGER_H*/
