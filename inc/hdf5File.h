/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_HDF5_FILE_H
#define EQ_MIVT_HDF5_FILE_H

#include <FileManager.h>

#include <hdf5.h>

namespace eqMivt
{
class hdf5File : public FileManager
{
	private:
		// HDF5 stuff
		hid_t           _file_id;
		hid_t           _dataset_id;
		hid_t           _spaceid;
		int             _ndim;
		hsize_t         _dims[3];

	public:

		virtual bool init(std::vector<std::string> file_params);

		~hdf5File();

		virtual void readCube(index_node_t index, float * cube, int levelCube, int nLevels, vmml::vector<3, int>    cubeDim, vmml::vector<3, int> cubeInc, vmml::vector<3, int> realCubeDim);
};
}

#endif /* EQ_MIVT_HDF5_FILE */
