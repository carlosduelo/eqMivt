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
		hid_t           file_id;
		hid_t           dataset_id;
		hid_t           spaceid;
		int             ndim;
		hsize_t         dims[3];

	public:

		hdf5File()
		{
		}

		virtual bool init(std::vector<std::string> file_params, int p_levelCube, int p_nLevels, vmml::vector<3, int> p_cubeDim, vmml::vector<3, int> p_cubeInc);

		~hdf5File();

		virtual void readCube(index_node_t index, float * cube);

		virtual vmml::vector<3, int> getRealDimension() { vmml::vector<3, int> dim; dim.set(dims[0],dims[1],dims[2]); return dim; };
};
}

#endif /* EQ_MIVT_HDF5_FILE */
