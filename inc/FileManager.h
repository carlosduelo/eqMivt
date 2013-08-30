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
		vmml::vector<3, int>	_offset;
		bool					_isInit;

	public:

		virtual ~FileManager() { _offset.set(0,0,0); _isInit = false;}

		bool isInit() { return _isInit; }

		void setOffset(vmml::vector<3, int> offset) { _offset = offset; }

		vmml::vector<3, int> getOffset() { return _offset; }

		virtual bool init(std::vector<std::string> file_params) = 0;

		virtual bool checkInit(std::string octree_file_name) = 0;

		virtual bool getxGrid(double ** xGrid) = 0;
		virtual bool getyGrid(double ** yGrid) = 0;
		virtual bool getzGrid(double ** zGrid) = 0;
		
		virtual void readCube(index_node_t index, float * cube, int levelCube, int nLevels, vmml::vector<3, int>    cubeDim, vmml::vector<3, int> cubeInc, vmml::vector<3, int> realCubeDim) = 0;

		virtual vmml::vector<3, int> getRealDimension() = 0;

};

}
#endif /* EQ_MIVT_FILE_MANAGER_H */
