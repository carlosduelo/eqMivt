/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <octreeContainer.h>

#include <octreeContainer_GPU.h>

#include <iostream>
#include <fstream>

namespace eqMivt
{

int OctreeContainer::getnLevelsFromOctreeFile(std::string file_name)
{
	/* Read octree from file */
	std::ifstream file;

	try
	{
		file.open(file_name.c_str(), std::ifstream::binary);
	}
	catch(...)
	{
		LBERROR<<"Octree: error opening octree file"<<std::endl;
		return 0;
	}

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		LBERROR<<"Octree: error invalid file format"<<std::endl;
		return 0;
	}

	int   no;
	float no1;
	int nLevels;

	file.read((char*)&no1, 	sizeof(float));
	file.read((char*)&no, 	sizeof(int));
	file.read((char*)&no, 	sizeof(int));
	file.read((char*)&no, 	sizeof(int));
	file.read((char*)&no, 	sizeof(int));
	file.read((char*)&nLevels, 	sizeof(int));

	return nLevels;
}

OctreeContainer::OctreeContainer(int device)
{
	_device = device;
	_octree = 0;
	_memoryGPU = 0;
	_sizes = 0;
}

OctreeContainer::~OctreeContainer()
{
	eqMivt::Destroy_OctreeContainer(_device, _octree, _memoryGPU, _sizes);
}

bool OctreeContainer::readOctreeFile(std::string file_name, int p_maxLevel)
{
	_maxLevel = p_maxLevel;

	/* Read octree from file */
	std::ifstream file;

	try
	{
		file.open(file_name.c_str(), std::ifstream::binary);
	}
	catch(...)
	{
		LBERROR<<"Octree: error opening octree file"<<std::endl;
		return false;
	}

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		LBERROR<<"Octree: error invalid file format"<<std::endl;
		return false;
	}

	file.read((char*)&_isosurface, 	sizeof(_isosurface));
	file.read((char*)&_dimension, 	sizeof(_dimension));
	file.read((char*)&_realDim[0], 	sizeof(_realDim[0]));
	file.read((char*)&_realDim[1], 	sizeof(_realDim[1]));
	file.read((char*)&_realDim[2], 	sizeof(_realDim[2]));
	file.read((char*)&_nLevels, 	sizeof(int));

	if (_maxLevel <= 0 || _maxLevel > _nLevels)
	{
		LBERROR<<"Octree: max level should be > 0 and < "<<_nLevels<<std::endl;
		return false;
	}

	LBINFO<<"Octree de dimension "<<_dimension<<"x"<<_dimension<<"x"<<_dimension<<" niveles "<<_nLevels<<std::endl;

	index_node_t ** octreeCPU       = new index_node_t*[_nLevels+1];
	int     *       sizesCPU        = new int[_nLevels+1];

	for(int i=_nLevels; i>=0; i--)
	{
		int numElem = 0;
		file.read((char*)&numElem, sizeof(numElem));
		//std::cout<<"Dimension del node en el nivel "<<i<<" es de "<<powf(2.0,*nLevels-i)<<std::endl;
		//std::cout<<"Numero de elementos de nivel "<<i<<" "<<numElem<<std::endl;
		sizesCPU[i] = numElem;
		octreeCPU[i] = new index_node_t[numElem];
		for(int j=0; j<numElem; j++)
		{
			index_node_t node = 0;
			file.read((char*) &node, sizeof(index_node_t));
			octreeCPU[i][j]= node;
		}
	}

	file.close();
	/* end reading octree from file */

	LBINFO<<"Copying octree to GPU"<<std::endl;

	std::string result;
	if (!eqMivt::Create_OctreeContainer(_device, octreeCPU, sizesCPU, _maxLevel, &_octree, &_memoryGPU, &_sizes, &result))
	{
		LBERROR<<"Octree: error creating octree in GPU"<<std::endl;
		return false;
	}
	LBINFO<<result;

	LBINFO<<"End copying octree to GPU"<<std::endl;

	delete[] sizesCPU;
	for(int i=0; i<=_nLevels; i++)
	{
		delete[] octreeCPU[i];
	}
	delete[]        octreeCPU;

	return true;
}


}
