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

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

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
	

	#ifndef DEBUG
	std::cout<<"Real Volume size:" <<_realDimVolume<<std::endl;
	std::cout<<"nLevels "<<_nLevels<<std::endl;
	std::cout<<"Level cube "<<_levelCube<<" size "<<_dimCube<<std::endl;
	std::cout<<"Start coordinate: "<<_startC<<std::endl;
	std::cout<<"Finish coordinate: "<<_finishC<<std::endl;
	std::cout<<"Real cubes size: "<<_sizeCube<<std::endl;
	#endif

	_isInit = true;

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

inline bool checkRange(index_node_t * elements, index_node_t index, int min, int max)
{
	return  index == elements[min] 	|| 
		index == elements[max]	||
		(elements[min] < index && elements[max] > index);
}

int mivtFile::getOffset(index_node_t index)
{
	bool end = false;
	bool found = false;
	int middle = 0;
	int min = 0;
	int max = _sizeNodes - 1;

	while(!end && !found)
	{
		int diff 	= max-min;
		middle	= min + (diff / 2);
		if (middle % 2 == 1) middle--;

		end 		= diff <= 1;
		found 		=  checkRange(_nodes, index, middle, middle+1);
		if (index < _nodes[middle])
			max = middle-1;
		else //(index > elements[middle+1])
			min = middle + 2;
	}

	if (found)
	{
		return _offsets[middle/2] + (index - _nodes[middle]);
	}
	else
		return -1;
}

void mivtFile::readCube(index_node_t index, float * cube, int levelCube, int nLevels, vmml::vector<3, int>    cubeDim, vmml::vector<3, int> cubeInc, vmml::vector<3, int> realCubeDim)
{
	if (!_isInit)
		return;

	vmml::vector<3, int> coord 	= getMinBoxIndex2(index, levelCube, nLevels);
	coord += _offset;
	vmml::vector<3, int> s 		= coord - cubeInc;
	vmml::vector<3, int> e 		= s + realCubeDim;

	int dim[3] = {abs(e.x()-s.x()),abs(e.y()-s.y()),abs(e.z()-s.z())};

	#ifdef DISK_TIMING 
	lunchbox::Clock     timing; 
	timing.reset();
	#endif
	// Set zeros's
	bzero(cube, dim[0]*dim[1]*dim[2]*sizeof(float));
	#ifdef DISK_TIMING 
	double time = timing.getTimed(); 
	std::cerr<<"Inicializate cube time: "<<time/1000.0<<" seconds."<<std::endl;
	timing.reset();
	#endif

	if (exp2(nLevels - levelCube) == exp2(_nLevels - _levelCube))
	{
		if (coord[0] % _dimCube == 0 && coord[1] % _dimCube == 0 && coord[2] % _dimCube == 0)
		{
			index_node_t idSearch = coordinateToIndex(coord, _levelCube, _nLevels); 
			
			int offset = getOffset(idSearch);

			_file.seekg(_startOffset + _sizeCube*offset*sizeof(float), std::ios_base::beg);
			_file.read((char*) cube, _sizeCube*sizeof(float));

			#ifndef DEBUG
			std::cout<<index<<" "<<idSearch<<" "<<coord<<" "<<offset<<" "<<_startOffset + _sizeCube*offset*sizeof(float)<<std::endl;

			if (_file)
				std::cout << "all characters read successfully.";
		    else
			      std::cout << "error: only " << _file.gcount() << " could be read";
			#endif
		}
		else
		{
			std::cerr<<"Not implemented: cube not aliegned"<<std::endl;
		}
	}
	else if (exp2(nLevels - levelCube) < exp2(_nLevels - levelCube))
	{
		if ( (coord[0] & (coord[0] -1 )) == 0 && (coord[1] & (coord[1] - 1 )) == 0 && (coord[2] & (coord[2] - 1 )) == 0)
		{
			index_node_t idSearch = coordinateToIndex(coord, _levelCube, _nLevels); 
			
			int offset = getOffset(idSearch);

			float * auxCube = new float[_sizeCube];
			_file.seekg(_startOffset + _sizeCube*offset*sizeof(float), std::ios_base::beg);
			_file.read((char*) auxCube, _sizeCube*sizeof(float));

			#ifndef DEBUG
			std::cout<<index<<" "<<idSearch<<" "<<coord<<" "<<offset<<" "<<_startOffset + _sizeCube*offset*sizeof(float)<<std::endl;
			
			if (_file)
				std::cout << "all characters read successfully.";
		    else
			      std::cout << "error: only " << _file.gcount() << " could be read";
			#endif

			int s = _dimCube + 2 * CUBE_INC;
			for(int i=0; i<realCubeDim.x(); i++)
				for(int j=0; j<realCubeDim.y(); j++)
					memcpy((void*) &cube[posToIndex(i, j, 0, realCubeDim.x())], (void*) &auxCube[posToIndex(coord[0]+i, coord[1]+j, coord[2], s)], realCubeDim.z()*sizeof(float));

			delete[] auxCube;
		}
		else
		{
			std::cerr<<"Not implemented: cube not aliegned"<<std::endl;
		}
	}
	else // if (exp2(nLevels - levelCube) < exp2(_nLevels - levelCube))
	{
		// NOT IMPLEMENTED
		std::cerr<<"Not implemented requested cube dimension > stored cube dimension"<<std::endl;
	}

	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Read in MB: "<<(dim[0]*dim[1]*dim[2]*sizeof(float)/1024.f/1024.f)<<" in "<<(time/1000.0f)<<" seconds."<<std::endl;
	std::cerr<<"Bandwidth: "<<(dim[0]*dim[1]*dim[2]*sizeof(float)/1024.f/1024.f)/(time/1000.0f)<<" MB/seconds."<<std::endl;
	#endif

	return;
}

vmml::vector<3, int> mivtFile::getRealDimension()
{
	return _realDimVolume; 
}
}
