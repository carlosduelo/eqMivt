/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <typedef.h>

#include <iostream>
#include <fstream>

int main( const int argc, char ** argv)
{
	if (argc < 2)
	{
		std::cerr<<"Please provaid octree file"<<std::endl;
		return 0;
	}

	/* Read octree from file */
	std::ifstream file;

	try
	{
		file.open(argv[1], std::ifstream::binary);
	}
	catch(...)
	{
		std::cerr<<"Octree: error opening octree file"<<std::endl;
		return 0;
	}

	int magicWord;
	float _isosurface;
	int _realDim[3];
	int _nLevels;
	int _dimension;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return false;
	}

	file.read((char*)&_isosurface, 	sizeof(_isosurface));
	file.read((char*)&_dimension, 	sizeof(_dimension));
	file.read((char*)&_realDim[0], 	sizeof(_realDim[0]));
	file.read((char*)&_realDim[1], 	sizeof(_realDim[1]));
	file.read((char*)&_realDim[2], 	sizeof(_realDim[2]));
	file.read((char*)&_nLevels, 	sizeof(int));

	for(int i=_nLevels; i>=0; i--)
	{
		std::cout<<"Level "<<i<<std::endl;
		int numElem = 0;
		file.read((char*)&numElem, sizeof(numElem));
		for(int j=0; j<numElem; j+=2)
		{
			eqMivt::index_node_t node = 0;
			file.read((char*) &node, sizeof(eqMivt::index_node_t));
			std::cout << "From "<<node;
			file.read((char*) &node, sizeof(eqMivt::index_node_t));
			std::cout << " to " << node<<std::endl;
		}
	}

	file.close();
	/* end reading octree from file */

	return 0;
}
