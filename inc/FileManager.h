/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_FILE_MANAGER_H
#define EQ_MIVT_FILE_MANAGER_H

#include <typedef.h>

#include <eq/eq.h>

namespace eqMivt
{

class FileManager
{
	public:

		virtual ~FileManager() { }

		virtual bool init(std::vector<std::string> file_params) = 0;
		
		virtual void readCube(index_node_t index, float * cube, int levelCube, int nLevels, vmml::vector<3, int>    cubeDim, vmml::vector<3, int> cubeInc, vmml::vector<3, int> realCubeDim) = 0;

};

}
#endif /* EQ_MIVT_FILE_MANAGER_H */
