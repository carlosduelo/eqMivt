/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <fileFactory.h>

namespace eqMivt
{
FileManager * CreateFileManage(std::string type_file, std::vector<std::string> file_params, int p_levelCube, int p_nLevels, vmml::vector<3, int> p_cubeDim, vmml::vector<3, int> p_cubeInc)
{
	if (type_file.compare("hdf5_file") == 0)
	{
		FileManager * hdf5 = new hdf5File();
		return hdf5->init(file_params, p_levelCube, p_nLevels, p_cubeDim, p_cubeInc) ? hdf5 : 0;
	}
	else
	{
		LBERROR<<"Error: wrong file option"<<std::endl;
	}

	return 0;
}
}

