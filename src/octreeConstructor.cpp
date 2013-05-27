/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "octreeConstructor.h"
#include "fileFactory.h"

namespace eqMivt
{
	bool octreeConstructor::createOctree(std::string type_file, std::vector<std::string> file_params, int maxLevel, std::vector<float> isosurfaceList, std::string octree_file)
	{
		vmml::vector<3, int> cubeDim(2,2,2);
		vmml::vector<3, int> cubeInc(0,0,0);
		FileManager * file = eqMivt::CreateFileManage(type_file, file_params, maxLevel, maxLevel, cubeDim, cubeInc); 
		vmml::vector<3, int> realDim = file->getRealDimension();
		delete file;

		int dimension;
		int nLevels;

		if (realDim[0]>realDim[1] && realDim[0]>realDim[2])
			dimension = realDim[0];
		else if (realDim[1]>realDim[2])
			dimension = realDim[1];
		else
			dimension = realDim[2];

		/* Calcular dimension del árbol*/
		float aux = logf(dimension)/logf(2.0);
		float aux2 = aux - floorf(aux);
		nLevels = aux2>0.0 ? aux+1 : aux;
		dimension = pow(2,nLevels);
	
		if (maxLevel > nLevels)
		{
			std::cerr<<"MaxLevel has to be <= "<<nLevels<<std::endl;
			return false;
		}

		int dimCube = pow(2,nLevels-maxLevel);
		cubeDim.set(dimCube, dimCube, dimCube);

		std::cout<<"Octree de dimension "<<dimension<<"x"<<dimension<<"x"<<dimension<<" niveles "<<nLevels<<std::endl;
		std::cout<<"Creating octree in file "<<octree_file<<std::endl;
		std::cout<<"Max level "<<maxLevel<<" cube dimension "<<dimCube<<std::endl;
		file = eqMivt::CreateFileManage(type_file, file_params, maxLevel, nLevels, cubeDim, cubeInc); 

		return true;
	}
}
