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
	protected:
		int			levelCube;
		int			nLevels;
		vmml::vector<3, int>	cubeDim;
		vmml::vector<3, int>	cubeInc;
		vmml::vector<3, int>	realCubeDim;

		bool			error;

	public:
		FileManager()
		{
			error = false;
		}

		virtual bool init(std::vector<std::string> file_params, int p_levelCube, int p_nLevels, vmml::vector<3, int> p_cubeDim, vmml::vector<3, int> p_cubeInc)
		{
			levelCube 	= p_levelCube;
			nLevels		= p_nLevels;
			cubeDim		= p_cubeDim;
			cubeInc		= p_cubeInc;
			realCubeDim	= p_cubeDim + 2*p_cubeInc;

			return true;
		}

		bool isError() { return error; }
		
		virtual void readCube(index_node_t index, float * cube) = 0;

		virtual ~FileManager() { }
};

}
#endif /* EQ_MIVT_FILE_MANAGER_H */
