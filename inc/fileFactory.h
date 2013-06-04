/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_FILE_FACTORY_H
#define EQ_MIVT_FILE_FACTORY_H

#include <hdf5File.h>
#include <vector>

namespace eqMivt
{
FileManager * CreateFileManage(std::string type_file, std::vector<std::string> file_params);
}
#endif /*EQ_MIVT_FILE_FACTORY_H*/
