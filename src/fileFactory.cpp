/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <fileFactory.h>

namespace eqMivt
{
FileManager * CreateFileManage(std::string type_file, std::vector<std::string> file_params)
{
	if (type_file.compare("hdf5_file") == 0)
	{
		FileManager * hdf5 = new hdf5File();
		return hdf5->init(file_params) ? hdf5 : 0;
	}
	else
	{
		LBERROR<<"Error: wrong file option"<<std::endl;
	}

	return 0;
}
}

