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
	_realDim = 0;
	_numOctrees = 0;
	_lastLevel = 0;
	_startC = 0;
	_finishC = 0;

	_xGrid = 0;
	_yGrid = 0;
	_zGrid = 0;

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
	if (_realDim != 0)
		delete[] _realDim;
	if (_startC != 0)
		delete[] _startC;
	if (_finishC != 0)
		delete[] _finishC;
	if (_xGrid != 0)
		delete[] _xGrid;
	if (_yGrid != 0)
		delete[] _yGrid;
	if (_zGrid != 0)
		delete[] _zGrid;
	if (_nLevels != 0)
		delete[] _nLevels;
	if (_maxLevel != 0)
		delete[] _maxLevel;
	if (_dimension != 0)
		delete[] _dimension;
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
		for(int i=0; i<=_lastLevel; i++)
			if (_octreeData[i] != 0)
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

	std::cout<<"Octree "<<_currentOctree<<std::endl;
	std::cout<<"nLevel "<<_nLevels[_currentOctree]<<std::endl;
	std::cout<<"maxLevel "<<_maxLevel[_currentOctree]<<std::endl;
	std::cout<<"dimension "<<_dimension[_currentOctree]<<std::endl;
	std::cout<<"Real Dim "<<_realDim[_currentOctree]<<std::endl;
	std::cout<<"Best Cube cache level "<<_cubeCacheLevel[_currentOctree]<<std::endl;;
	std::cout<<"Coordinates from "<<_startC[_currentOctree]<<" to "<<_finishC[_currentOctree]<<std::endl;


	_file.seekg(_desp[0], std::ios_base::beg);
	for(int d=1; d<=_currentOctree; d++)
		_file.seekg(_desp[d], std::ios_base::cur);

	_file.seekg(((2*(_maxLevel[_currentOctree]+1))+1)*sizeof(int), std::ios_base::cur);
	if (_octreeData == 0)
	{
		_octreeData = new index_node_t*[_lastLevel+1];
		for(int i=0; i<=_lastLevel; i++)
			_octreeData[i] = 0;
	}

	for(int i=0; i<=_maxLevel[_currentOctree]; i++)
	{
		if (_octreeData[i] != 0)
			delete[] _octreeData[i];
		_octreeData[i] = new index_node_t[_sizes[_currentOctree][i]];
		_file.read((char*)_octreeData[i], _sizes[_currentOctree][i]*sizeof(index_node_t));
	}
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

	_file.read((char*)&_numOctrees,sizeof(_numOctrees));
	_file.read((char*)_realDimensionVolume.array,3*sizeof(int));
	_xGrid = new double[_realDimensionVolume.x()];
	_yGrid = new double[_realDimensionVolume.y()];
	_zGrid = new double[_realDimensionVolume.z()];
	_file.read((char*)_xGrid,_realDimensionVolume.x()*sizeof(double));
	_file.read((char*)_yGrid,_realDimensionVolume.y()*sizeof(double));
	_file.read((char*)_zGrid,_realDimensionVolume.z()*sizeof(double));

	_nLevels = new int[_numOctrees];
	_maxLevel = new int[_numOctrees];
	_dimension = new int[_numOctrees];
	_realDim = new vmml::vector<3, int>[_numOctrees];
	_startC = new vmml::vector<3, int>[_numOctrees];
	_finishC= new vmml::vector<3, int>[_numOctrees];

	int rest = 0;
	while(rest < _numOctrees)
	{
		int n = 0;
		int nL = 0;
		int mL = 0;
		vmml::vector<3, int> s;
		vmml::vector<3, int> f;
		vmml::vector<3, int> d;
		_file.read((char*)&n,sizeof(int));
		_file.read((char*)&s.array,3*sizeof(int));
		_file.read((char*)&f.array,3*sizeof(int));
		_file.read((char*)&nL,sizeof(int));
		_file.read((char*)&mL,sizeof(int));
		d = f - s;
		for(int j=0; j<n; j++)
		{
			_nLevels[rest+j] = nL;
			_maxLevel[rest+j] = mL;
			_dimension[rest+j] = exp2(nL);
			_startC[rest+j] = s;
			_finishC[rest+j] = f;
			_realDim[rest+j] = d;
		}
		rest += n;
	}

	_lastLevel = 0;
	for(int i=0; i<_numOctrees; i++)
		_lastLevel = _maxLevel[i] > _lastLevel ? _maxLevel[i] : _lastLevel;


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
		_numCubes[i] = new int[_maxLevel[i] + 1];
		_sizes[i] = new int[_maxLevel[i] + 1];
		_file.seekg(_desp[0], std::ios_base::beg);
		for(int d=1; d<=i; d++)
			_file.seekg(_desp[d], std::ios_base::cur);
		_file.read((char*)&_maxHeight[i], sizeof(int));
		_file.read((char*)_numCubes[i], (_maxLevel[i]+1)*sizeof(int));
		_file.read((char*)_sizes[i], (_maxLevel[i]+1)*sizeof(int));
	}

	_setBestCubeLevel();


	return true;

}

void OctreeManager::_setBestCubeLevel()
{
	for(int i=0; i<_numOctrees; i++)
	{
		int mL = _nLevels[i] - 9 + 1; 
		if (mL <= 0)
			mL = 1;
		_cubeCacheLevel[i] = mL;//_maxLevel;
	}
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
		result = it->second->setCurrentOctree(_realDim[_currentOctree], _dimension[_currentOctree], _nLevels[_currentOctree], _maxLevel[_currentOctree], _cubeCacheLevel[_currentOctree], _isosurfaces[_currentOctree],  _maxHeight[_currentOctree], _octreeData, _sizes[_currentOctree], _lastLevel);
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
		o->setGeneralValues(device);
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
