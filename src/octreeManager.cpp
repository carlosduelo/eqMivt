/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "octreeManager.h"


namespace eqMivt
{
		OctreeManaer::OctreeManager()
		{
			_nLevels = 0;
			_maxLevel = 0;
			_dimension = 0;
			_realDim.set(0,0,0);
			_numOctrees = 0;

			_isosurfaces = 0;
			_sizes = 0;
			_numCubes = 0;
			_maxHeight = 0;
			_desp = 0;
			_cubeCacheLevel = 0;
			_octreeData = 0;
			_currentOctree = 0;
		}

		OctreeManaer::~OctreeManager()
		{
			if (_isosurfaces!=0)
				delete[] _isosurfaces;
			if (_numCubes!=0)
				delete[] _numCubes;
			if (_desp!=0)
				delete[] _desp;
			if (_cubeCacheLevel!=0)
				delete[] _cubeCacheLevel;
			if (_sizes!=0)
			{
				for(int i=0; i<_numOctrees; i++)
					delete[] _sizes[i];
				delete[] _sizes;
			}
			if (_octreeData!=0)
			{
				for(int i=0; i<=_maxLevel; i++)
					delete[] _octreeData[i];
				delete[] _octreeData;
			}

			_file.close();
		}

		bool OctreeManager::_checkOctreeFileValid(std::ifstream * file)
		{
			int magicWord;

			file->read((char*)&magicWord, sizeof(magicWord));

			if (magicWord != 919278872)
			{
				std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
				return false;
			}
			return true;
		}

		void OctreeManager::_readCurrentOctree()
		{
			_file.seekg(_desp[0], std::ios_base::beg);
			for(int d=1; d<=_numOctrees; d++)
				_file.seekg(_desp[d], std::ios_base::cur);

			file->seekg(((2*(maxLevel+1))+1)*sizeof(int), std::ios_base::cur);
			for(int i=0; i<=maxLevel; i++)
			{
				_octreeData[i] = new eqMivt::index_node_t[_sizes[i]];
				_file.read((char*)_octreeData[i], _sizes[i]*sizeof(eqMivt::index_node_t));
			}

		}

		int OctreeManaer::readNLevelsFromFile(std::string file_name)
		{
			std::ifstream file;

			try
			{
				file.open(file_name.c_str(), std::ifstream::binary);
			}
			catch(...)
			{
				std::cerr<<"Octree: error opening octree file"<<std::endl;
				return -1;
			}
			if (!_checkOctreeFileValid(&file))
			{
				return -1;
			}
			int nLevels = 0;

			file.read((char*)&nLevels,sizeof(nLevels));

			return nLevels;

		}
		int OctreeManaer::readMaxLevelsFromFile(std::string file_name)
		{
			std::ifstream file;

			try
			{
				file.open(file_name.c_str(), std::ifstream::binary);
			}
			catch(...)
			{
				std::cerr<<"Octree: error opening octree file"<<std::endl;
				return -1;
			}
			if (!_checkOctreeFileValid(&file))
			{
				return -1;
			}
			int maxLevel = 0;

			file.seekg(sizeof(int), std::ios_base::cur);
			file.read((char*)&maxLevel,sizeof(maxLevel));
			
			return maxLevel;

		}
		int OctreeManaer::readDimensionFromFile(std::string file_name)
		{
			std::ifstream file;

			try
			{
				file.open(file_name.c_str(), std::ifstream::binary);
			}
			catch(...)
			{
				std::cerr<<"Octree: error opening octree file"<<std::endl;
				return -1;
			}
			if (!_checkOctreeFileValid(&file))
			{
				return -1;
			}

			int dimension = 0;

			file.seekg(2*sizeof(int), std::ios_base::cur);
			file.read((char*)&dimension,sizeof(dimension));

			return dimension;

		}

		vmml::vector<3, int> OctreeManaer::readRealDimFromFile(std::string file_name)
		{
			std::ifstream file;

			try
			{
				file.open(file_name.c_str(), std::ifstream::binary);
			}
			catch(...)
			{
				std::cerr<<"Octree: error opening octree file"<<std::endl;
				return -1;
			}
			if (!_checkOctreeFileValid(&file))
			{
				return -1;
			}
			vmml::vector<3, int> realDim(0, 0, 0);

			file.seekg(3*sizeof(int), std::ios_base::cur);
			file.read((char*)realDim.array(),3*sizeof(int));

			return realDim;

		}
		int OctreeManaer::readNumOctreesFromFile(std::string file_name)
		{
			std::ifstream file;

			try
			{
				file.open(file_name.c_str(), std::ifstream::binary);
			}
			catch(...)
			{
				std::cerr<<"Octree: error opening octree file"<<std::endl;
				return -1;
			}
			if (!_checkOctreeFileValid(&file))
			{
				return -1;
			}
			int numOctrees = 0;

			file.seekg(6*sizeof(int), std::ios_base::cur);
			file.read((char*)&numOctrees,sizeof(numOctrees));

			return numOctrees;
		}

		bool OctreeManaer::init(std::string file_name);
		{
			try
			{
				_file.open(file_name.c_str(), std::ifstream::binary);
			}
			catch(...)
			{
				std::cerr<<"Octree: error opening octree file"<<std::endl;
				return false;
			}
			if (!_checkOctreeFileValid(&file))
			{
				return false;
			}

			_file.read((char*)&_nLevels,sizeof(nLevels));
			_file.read((char*)&_maxLevel,sizeof(maxLevel));
			_file.read((char*)&_dimension,sizeof(dimension));
			_file.read((char*)_realDim.array,3*sizeof(int));
			_file.read((char*)&_numOctrees,sizeof(numOctrees));

			_isosurfaces = new float[_numOctrees];	
			_file.read((char*)_isosurfaces, numOctrees*sizeof(float));
			_desp = new int[_numOctrees];
			_file.read((char*)_desp, numOctrees*sizeof(int));

			_numCubes = new int*[_numOctrees];
			_sizes = new int*[_numOctrees];
			_maxHeight = new int[_numOctrees];
			for(int i=0; i<_numOctrees; i++)
			{
				_numCubes[i] = new int[_maxLevel + 1];
				_sizes[i] = new int[_maxLevel + 1];
				_file.seekg(_desp[0], std::ios_base::beg);
				for(int d=1; d<=i; d++)
					_file.seekg(_desp[d], std::ios_base::cur);
				_file.read((char*)&_maxHeight[i], sizeof(int));
				_file.read((char*)_numCubes[i], (_maxLevel+1)*sizeof(int));
				_file.read((char*)_sizes[i], (_maxLevel+1)*sizeof(int));
			}

			return true;

		}
#if 0
		void OctreeManaer::setCurrentOctree(int currentOctree);

		void OctreeManaer::cpyOctreeToDevice(int device);
#endif
}
