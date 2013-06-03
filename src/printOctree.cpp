/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <typedef.h>

#include <iostream>
#include <fstream>

void readOctree(std::ifstream * file, int numO, int * desp, int * sizes, eqMivt::index_node_t ** octree, int maxLevel)
{
	file->seekg(desp[0], std::ios_base::beg);
	for(int d=1; d<=numO; d++)
		file->seekg(desp[d], std::ios_base::cur);

	file->seekg(((2*(maxLevel+1))+1)*sizeof(int), std::ios_base::cur);
	for(int i=0; i<=maxLevel; i++)
	{
		octree[i] = new eqMivt::index_node_t[sizes[i]];
		file->read((char*)octree[i], sizes[i]*sizeof(eqMivt::index_node_t));
	}
	
}

void printOctree(eqMivt::index_node_t ** octree, int * sizes, int maxLevel)
{
	for(int i=0; i<=maxLevel; i++)
	{
		std::cout	<<"Level: "<<i<<std::endl;
		for(int j=0; j<sizes[i]; j+=2)
		{
			std::cout << "From " <<octree[i][j]<<" to " << octree[i][j+1]<<std::endl;
		}
	}
}

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
		file.seekg(desp[0], std::ios_base::beg);
		for(int d=1; d<=i; d++)
			file.seekg(desp[d], std::ios_base::cur);
		file.read((char*)&maxHeight[i], sizeof(int));
		file.read((char*)numCubes[i], (maxLevel+1)*sizeof(int));
		file.read((char*)sizes[i], (maxLevel+1)*sizeof(int));
	}

	int selection = -1;
	while(selection < 0 || selection >= numOctrees) 
	{
		std::cout<<"Select octree to print [0,"<<numOctrees-1<<"] ";
		std::cin>>selection;
	}

	eqMivt::index_node_t ** octree;
	octree = new eqMivt::index_node_t*[maxLevel+1];

	readOctree(&file, selection, desp, sizes[selection], octree, maxLevel);

	printOctree(octree, sizes[selection], maxLevel);

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
	for(int i=0; i<=maxLevel; i++)
		delete[] octree[i];

	return 0;
}
