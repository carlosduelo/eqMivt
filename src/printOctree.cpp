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

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return false;
	}
	
	int nLevels = 0;
	int maxLevel = 0;
	int dimension = 0;
	int realDim[3] = {0, 0, 0};
	int numOctrees = 0;

	file.read((char*)&nLevels,sizeof(nLevels));
	file.read((char*)&maxLevel,sizeof(maxLevel));
	file.read((char*)&dimension,sizeof(dimension));
	file.read((char*)&realDim[0],sizeof(realDim[0]));
	file.read((char*)&realDim[1],sizeof(realDim[1]));
	file.read((char*)&realDim[2],sizeof(realDim[2]));
	file.read((char*)&numOctrees,sizeof(numOctrees));

	std::cout<<"Real dimension: "<<realDim[0]<<"x"<<realDim[1]<<"x"<<realDim[2]<<std::endl;
	std::cout<<"Dimension octree: "<<dimension<<"x"<<dimension<<"x"<<dimension<<" levels "<<nLevels<<std::endl;
	std::cout<<"Max level: "<<maxLevel<<std::endl;
	std::cout<<"Num octrees: "<<numOctrees<<std::endl;

	std::cout<<"Isosurfaces availables:"<<std::endl;
	float * isos = new float[numOctrees];
	file.read((char*)isos, numOctrees*sizeof(float));
	for(int i=0; i<numOctrees; i++)
		std::cout<<i<<". "<<isos[i]<<std::endl; 

	int * desp	= new int[numOctrees];
	file.read((char*)desp, numOctrees*sizeof(int));

	int ** numCubes = new int*[numOctrees];
	int ** sizes = new int*[numOctrees];
	int	* maxHeight = new int[numOctrees];
	for(int i=0; i<numOctrees; i++)
	{
		numCubes[i] = new int[maxLevel + 1];
		sizes[i] = new int[maxLevel + 1];
		std::cout<<desp[i]<<std::endl;
		file.seekg(desp[0], std::ios_base::beg);
		for(int d=1; d<=i; d++)
			file.seekg(desp[d], std::ios_base::cur);
		file.read((char*)&maxHeight[i], sizeof(int));
		std::cout<<i<<" ------> Max Height "<<maxHeight[i]<<std::endl;
		file.read((char*)numCubes[i], (maxLevel+1)*sizeof(int));
		file.read((char*)sizes[i], (maxLevel+1)*sizeof(int));
		for(int j=0; j<=maxLevel; j++)
			std::cout<<"Level: "<<j<<" "<<numCubes[i][j]<<" "<<sizes[i][j]<<std::endl;
	}


	file.close();
	/* end reading octree from file */

	delete[] isos;
	delete[] desp;
	delete[] maxHeight;
	for(int i=0; i<numOctrees; i++)
	{
		delete[] numCubes[i];
		delete[] sizes[i];
	}
	delete[] sizes;
	delete[] numCubes;

	return 0;
}
