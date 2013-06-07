/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "octreeManager.h"


namespace eqMivt
{
OctreeManager::OctreeManager()
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
	_currentOctree = -1;
}

OctreeManager::~OctreeManager()
{
	if (_isosurfaces!=0)
		delete[] _isosurfaces;
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
	if (_numCubes!=0)
	{
		for(int i=0; i<_numOctrees; i++)
			delete[] _numCubes[i];
		delete[] _numCubes;
	}
	if (_octreeData!=0)
	{
		for(int i=0; i<=_maxLevel; i++)
			delete[] _octreeData[i];
		delete[] _octreeData;
	}

	_file.close();

	for(std::map<uint32_t , Octree *>::iterator it=_octrees.begin(); it!=_octrees.end(); it++)
	{
		delete it->second;	
	}
}

void OctreeManager::_readCurrentOctree()
{
	_file.seekg(_desp[0], std::ios_base::beg);
	for(int d=1; d<=_currentOctree; d++)
		_file.seekg(_desp[d], std::ios_base::cur);

	_file.seekg(((2*(_maxLevel+1))+1)*sizeof(int), std::ios_base::cur);
	if (_octreeData == 0)
	{
		_octreeData = new index_node_t*[_maxLevel+1];
		for(int i=0; i<=_maxLevel; i++)
			_octreeData[i] = 0;
	}

	for(int i=0; i<=_maxLevel; i++)
	{
		if (_octreeData[i] != 0)
			delete[] _octreeData[i];
		_octreeData[i] = new index_node_t[_sizes[_currentOctree][i]];
		_file.read((char*)_octreeData[i], _sizes[_currentOctree][i]*sizeof(index_node_t));
	}

}

int OctreeManager::readNLevelsFromFile(std::string file_name)
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

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return -1;
	}

	int nLevels = 0;

	file.read((char*)&nLevels,sizeof(nLevels));

	return nLevels;

}
int OctreeManager::readMaxLevelsFromFile(std::string file_name)
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

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return -1;
	}
	int maxLevel = 0;

	file.seekg(sizeof(int), std::ios_base::cur);
	file.read((char*)&maxLevel,sizeof(maxLevel));
	
	return maxLevel;

}
int OctreeManager::readDimensionFromFile(std::string file_name)
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

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return -1;
	}

	int dimension = 0;

	file.seekg(2*sizeof(int), std::ios_base::cur);
	file.read((char*)&dimension,sizeof(dimension));

	return dimension;

}

vmml::vector<3, int> OctreeManager::readRealDimFromFile(std::string file_name)
{
	std::ifstream file;

	try
	{
		file.open(file_name.c_str(), std::ifstream::binary);
	}
	catch(...)
	{
		std::cerr<<"Octree: error opening octree file"<<std::endl;
		vmml::vector<3, int> r(-1,-1,-1);
		return r;
	}

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		vmml::vector<3, int> r(-1,-1,-1);
		return r;
	}
	vmml::vector<3, int> realDim(0, 0, 0);

	file.seekg(3*sizeof(int), std::ios_base::cur);
	file.read((char*)realDim.array,3*sizeof(int));

	return realDim;

}
int OctreeManager::readNumOctreesFromFile(std::string file_name)
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

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return -1;
	}
	int numOctrees = 0;

	file.seekg(6*sizeof(int), std::ios_base::cur);
	file.read((char*)&numOctrees,sizeof(numOctrees));

	return numOctrees;
}

bool OctreeManager::init(std::string file_name)
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
	int magicWord;

	_file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return false;
	}

	_file.read((char*)&_nLevels,sizeof(_nLevels));
	_file.read((char*)&_maxLevel,sizeof(_maxLevel));
	_file.read((char*)&_dimension,sizeof(_dimension));
	_file.read((char*)_realDim.array,3*sizeof(int));
	_file.read((char*)&_numOctrees,sizeof(_numOctrees));

	_isosurfaces = new float[_numOctrees];	
	_file.read((char*)_isosurfaces, _numOctrees*sizeof(float));
	_desp = new int[_numOctrees];
	_file.read((char*)_desp, _numOctrees*sizeof(int));

	_numCubes = new int*[_numOctrees];
	_sizes = new int*[_numOctrees];
	_maxHeight = new int[_numOctrees];
	_cubeCacheLevel = new int[_numOctrees];
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

	_setBestCubeLevel();

	return true;

}

void OctreeManager::_setBestCubeLevel()
{
	for(int i=0; i<_numOctrees; i++)
		_cubeCacheLevel[i] = 3;//_maxLevel;
}

bool OctreeManager::setCurrentOctree(int currentOctree)
{
	_lock.set();
	
	bool result = true;

	if (currentOctree != _currentOctree)
	{
		_currentOctree = currentOctree;

		try
		{
			_readCurrentOctree();
		}
		catch(...)
		{
			std::cerr<<"Error reading octree from file"<<std::endl;
			result = false;
		}
	}

	_lock.unset();

	return result;
}

bool OctreeManager::checkStatus(uint32_t device)
{
	_lock.set();
	bool result = true;
	std::map<uint32_t, Octree *>::iterator it;
	it = _octrees.find(device);

	if (it == _octrees.end())
	{
		result = false;
	}
	else
	{
		result = it->second->setCurrentOctree(_cubeCacheLevel[_currentOctree], _isosurfaces[_currentOctree],  _maxHeight[_currentOctree], _octreeData, _sizes[_currentOctree]);
	}

	_lock.unset();

	return result;
}

Octree * OctreeManager::getOctree(uint32_t  device)
{
	_lock.set();

	Octree * o = 0;

	std::map<uint32_t, Octree *>::iterator it;
	it = _octrees.find(device);

	// Not exist create
	if (it == _octrees.end())
	{
		o = new Octree();
		o->setGeneralValues(_realDim, _dimension, _nLevels, _maxLevel, device);
		#if 0
		if (o->setCurrentOctree(_isosurfaces[_currentOctree],  _maxHeight[_currentOctree], _octreeData, _sizes[_currentOctree]))
		{
			_octrees[device] = o;
		}
		else
		{
			delete o;
			o = 0;
		}
		#else
		_octrees[device] = o;
		#endif
	}
	else
	{
		o = it->second;
		#if 0
		if (!o->setCurrentOctree(_isosurfaces[_currentOctree],  _maxHeight[_currentOctree], _octreeData, _sizes[_currentOctree]))
		{
			o = 0;
		}
		#endif
	}
	_lock.unset();

	return o;

}


}
