/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <mivtFile.h>

#include <mortonCodeUtil_CPU.h>

#include <iostream>
#include <strings.h>

#ifdef DISK_TIMING
#include <lunchbox/clock.h>
#endif

namespace eqMivt
{
bool mivtFile::init(std::vector<std::string> file_params)
{

	try
	{
		_file.open(file_params[0].c_str(), std::ifstream::binary);
	}
	catch(...)
	{
		std::cerr<<"FileManager: error opening data file"<<std::endl;
		return false;
	}

	_file.read((char*) _md5sum, MD5_DIGEST_LENGTH*sizeof(unsigned char));
	_file.read((char*) _realDimVolume.array, 3*sizeof(int));
	_file.read((char*) &_nLevels, sizeof(int));
	_file.read((char*) &_levelCube, sizeof(int));
	_file.read((char*) _startC.array, 3*sizeof(int));
	_file.read((char*) _finishC.array, 3*sizeof(int));

	_file.read((char*) &_sizeNodes, sizeof(int));
	
	_nodes = 0;
	_offsets = 0;
	_nodes = new index_node_t[_sizeNodes];
	_offsets = new int[_sizeNodes/2];

	try
	{
		_file.read((char*) _nodes, _sizeNodes*sizeof(index_node_t));
		_file.read((char*) _offsets, (_sizeNodes/2)*sizeof(int));
	}
	catch (...)
	{
		std::cerr<<"FileManager: error opening data file"<<std::endl;
		return false;
	}

	_startOffset = MD5_DIGEST_LENGTH*sizeof(unsigned char) + (12 + (_sizeNodes/2))*sizeof(int) + _sizeNodes*sizeof(index_node_t);

	_dimCube = exp2(_nLevels - _levelCube);
	_sizeCube = powf(_dimCube + 2 * CUBE_INC, 3);

	return true;
}

bool mivtFile::checkInit(std::string octree_file_name)
{
	unsigned char md5sum[MD5_DIGEST_LENGTH];
	FILE * inFile = fopen (octree_file_name.c_str(), "rb");
	MD5_CTX mdContext;
	int bytes;
	unsigned char data[1024];
	MD5_Init (&mdContext);
	while ((bytes = fread (data, 1, 1024, inFile)) != 0)
		MD5_Update (&mdContext, data, bytes);
	MD5_Final(md5sum, &mdContext);

	bool result = true;
	for(int i = 0; i < MD5_DIGEST_LENGTH; i++)
		if (md5sum[i] != _md5sum[i])
		{
			result = false;
			std::cerr<<"Error this mivt file is not valid for "<<octree_file_name<<" octree file"<<std::endl;
			break;
		}

	fclose (inFile);
	return result;
}

mivtFile::~mivtFile()
{
	if (_nodes != 0)
		delete[] _nodes;
	if (_offsets != 0)
		delete[] _offsets;
	_file.close();
}

bool mivtFile::getxGrid(double ** xGrid)
{
	return false;
}

bool mivtFile::getyGrid(double ** yGrid)
{
	return false;
}

bool mivtFile::getzGrid(double ** zGrid)
{
	return false;
}


void mivtFile::readCube(index_node_t index, float * cube, int levelCube, int nLevels, vmml::vector<3, int>    cubeDim, vmml::vector<3, int> cubeInc, vmml::vector<3, int> realCubeDim)
{
	return;
}

vmml::vector<3, int> mivtFile::getRealDimension()
{
	return _realDimVolume; 
}
}
